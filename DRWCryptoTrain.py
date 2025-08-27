import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

import lightgbm as lgb
import xgboost as xgb

from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge

from statsmodels.tsa.stattools import adfuller, kpss

class DRWCryptoTrain:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.features = pd.DataFrame(index=self.df.index)
        self.X_train = None
        self.y_train = None
        self.X_eval = None
        self.y_eval = None
        self.stationary_df = None
        self.corr_matrix = None
        self.train_prediction_df = None
        self.eval_prediction_df = None
        self.pca = None
        self.standard_scaler = None

    def create_market_microstructure_features(self):
        d = self.df
        v = d['volume']
        bq, sq = d['buy_qty'], d['sell_qty']
        bid, ask = d['bid_qty'], d['ask_qty']
        eps = 1e-9

        self.features['other_qty'] = v - (bq + sq)

        # 1. Liquidity features
        self.features['liquidity_imbalance'] = ask - bid
        self.features['bid_ask_ratio'] = bid / (ask + eps)
        self.features['total_liquidity'] = bid + ask
        self.features['norm_liquidity_imbalance'] = self.features['liquidity_imbalance'] / (ask + bid + eps)

        # 2. Order flow
        self.features['net_order_flow'] = bq - sq
        self.features['buy_sell_ratio'] = bq / (sq + eps)
        self.features['total_order_flow'] = bq + sq
        self.features['order_book_participation'] = (bq + sq) / (v + eps)
        self.features['norm_net_order_flow'] = self.features['net_order_flow'] / (bq + sq + eps)

        # 3. Price pressure
        self.features['buy_pressure'] = bq / (v + eps)
        self.features['sell_pressure'] = sq / (v + eps)
        self.features['net_pressure'] = self.features['buy_pressure'] - self.features['sell_pressure']
        self.features['pressure_ratio'] = self.features['buy_pressure'] / (self.features['sell_pressure'] + eps)

        # 4. Market depth/width ratios
        self.features['depth_ratio'] = (bid + ask) / (v + eps)
        self.features['bid_depth_ratio'] = bid / (v + eps)
        self.features['ask_depth_ratio'] = ask / (v + eps)

        # 5. Microstructure impact metrics
        self.features['kyle_lambda_proxy'] = np.abs(bq - sq) / (v + eps)
        self.features['liquidity_consumption'] = v / ((bid + ask) + eps)

        # 6. Market quality proxies
        self.features['execution_quality_proxy'] = v / (self.features['liquidity_imbalance'].abs() + 1)

        # 7. Information toxicity
        self.features['order_toxicity'] = np.abs(self.features['norm_net_order_flow']) * self.features['kyle_lambda_proxy']

        # 8. Interaction effects
        self.features['bid_momentum'] = bid * bq / (v + eps)
        self.features['ask_momentum'] = ask * sq / (v + eps)

        print('Feature engineering completed.')
        print("="*25)

    def split_train_eval(self, val_size=86400): # 2-month of data
        idx = self.df.index

        train_idx = idx[:-val_size]
        eval_idx = idx[-val_size:]

        self.y_train = self.df.loc[train_idx, 'label']
        self.y_eval = self.df.loc[eval_idx, 'label']

        X_all = pd.concat([self.df.drop(columns=['label']), self.features], axis=1)
        self.X_train = X_all.loc[train_idx].copy()
        self.X_eval = X_all.loc[eval_idx].copy()

        print(f"Train-Eval split completed: {len(train_idx)} rows for training, {len(eval_idx)} rows for evaluation")
        print("="*25)

    def standardize_features_train_eval(self):
        cols = [col for col in self.X_train.columns if 'label' not in col]
        scaler_df = StandardScaler()
        self.X_train[cols] = scaler_df.fit_transform(self.X_train[cols])
        self.X_eval[cols] = scaler_df.transform(self.X_eval[cols])

        self.standard_scaler = scaler_df

        print("Standardizing features")
        print("="*25)

    def check_standardised_features(self):
        means = self.X_train.mean()
        stds = self.X_train.std()

        print("Mean across X-features (abs avg):", means.abs().mean())
        print("Std across X-features (avg):", stds.mean())
        print("="*25)

    def analyse_variance(self):
        selector = VarianceThreshold(threshold=0.0)
        selector.fit(self.X_train)

        support_mask = selector.get_support()
        kept_columns = self.X_train.columns[support_mask]
        dropped_columns = self.X_train.columns[~support_mask]
        print("Dropped zero-variance columns:", list(dropped_columns))
        print("="*25)

        self.X_train = self.X_train.loc[:, support_mask]

    def explore_target_variable(self):
        print("=== Target Statistics ===")
        print(f"Mean: {self.y_train.mean():.6f}")
        print(f"Std: {self.y_train.std():.6f}")
        print(f"Skewness: {self.y_train.skew():.2f}")
        print(f"Kurtosis: {self.y_train.kurtosis():.2f}")
        print(f"Min: {self.y_train.min():.6f}, Max: {self.y_train.max():.6f}")
        print(f"Negative values: {(self.y_train < 0).sum()} | Zero values: {(self.y_train == 0).sum()} | Positive values: {(self.y_train > 0).sum()}")

        # Plot distribution
        plt.figure(figsize=(8, 4))
        sns.histplot(self.y_train, bins=100, kde=True)
        plt.title("Target Variable Distribution (Price Change)")
        plt.xlabel("Price Change")
        plt.tight_layout()
        plt.show()

        # Stationarity check with ADF test
        result = adfuller(self.y_train.dropna(), maxlag=10, autolag=None)
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        print("=> Stationary" if result[1] < 0.05 else "=> Non-Stationary")
        print("="*25)

    def generate_correlation_matrix(self, filename, corr_method):
        output_path = f'inputs/{filename}_{corr_method}.csv'
        if os.path.exists(output_path):
            self.corr_matrix = pd.read_csv(output_path, index_col=0)
            print(f"Loaded existing correlation matrix from {output_path}")
        else:
            corr = self.X_train.corr(method=corr_method).abs()
            corr.to_csv(output_path)
            self.corr_matrix = corr
            print(f"Computed and saved correlation matrix to {output_path}")
        print("="*25)

    def check_time_series_stability(self, filename):
        output_path = f'inputs/{filename}.csv'
        if os.path.exists(output_path):
            self.stationary_df = pd.read_csv(output_path)
            print(f"Loaded exisiting stationary df from {output_path}")
        else: 
            def adf_pval(series):
                try:
                    return adfuller(series.dropna(), maxlag=10, autolag=None)[1]
                except:
                    return np.nan

            def kpss_pval(series):
                try:
                    return kpss(series.dropna(), regression='c', nlags='auto')[1]
                except:
                    return np.nan

            results = []
            for col in self.X_train.columns:
                s = self.X_train[col]
                results.append({
                    'feature': col,
                    'adf_pval': adf_pval(s),
                    'kpss_pval': kpss_pval(s),
                })

            self.stationary_df = pd.DataFrame(results)
            self.stationary_df['adf_stationary'] = self.stationary_df['adf_pval'] < 0.05
            self.stationary_df['kpss_stationary'] = self.stationary_df['kpss_pval'] >= 0.05
            self.stationary_df.to_csv(output_path, index=False)
            print(f"Computed and saved stationary df to {output_path}")
        print("="*25)
    
    def pps_lightgbm_regression(self, X_pps, y_pps, n_splits=4, test_size=None, gap=0, lgb_params=None, random_state=42):
        X = np.array(X_pps).reshape(-1, 1)
        y = np.array(y_pps.astype('float32'))
        
        params = lgb_params or {
            'objective': 'regression',
            'metric': 'l2',
            'verbosity': -1,
            'verbose': -1,
            'seed': random_state,
        }

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
        
        errors = []
        for train_idx, val_idx in tscv.split(X):
            dtrain = lgb.Dataset(X[train_idx], label=y[train_idx])
            dval   = lgb.Dataset(X[val_idx],   label=y[val_idx], reference=dtrain)

            callbacks = [
                lgb.early_stopping(stopping_rounds=10, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0)
            ]
            
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=100,
                callbacks=callbacks
            )
            
            preds = model.predict(X[val_idx], num_iteration=model.best_iteration)
            errors.append(mean_squared_error(y[val_idx], preds))
        
        model_mse = np.mean(errors)
        baseline_var = np.var(y)
        pps = max(0, 1 - model_mse / baseline_var)
        return pps

    def generate_predictive_power_df(self, filename, n_splits=4, test_size=None, gap=0, lgb_params=None, random_state=42):
        output_path = f'inputs/{filename}.csv'
        if os.path.exists(output_path):
            self.predictive_power_df = pd.read_csv(output_path)
            print(f"Loaded exisiting predictive power df from {output_path}")
        else:
            records = []
            pearson_corr = self.X_train.corrwith(self.y_train, method='pearson')
            spearman_corr = self.X_train.corrwith(self.y_train, method='spearman')
            mi_scores = mutual_info_regression(self.X_train, self.y_train, random_state=random_state)
            mi_series = pd.Series(mi_scores, index=self.X_train.columns)

            for feature in self.X_train.columns:
                feat_series = self.X_train[feature].dropna()
                pps_score = self.pps_lightgbm_regression(feat_series, self.y_train, n_splits=n_splits,
                                        test_size=test_size, gap=gap, lgb_params=lgb_params)
                records.append({
                    'feature': feature,
                    'pps': pps_score,
                    'pearson': pearson_corr[feature],
                    'spearman': spearman_corr[feature],
                    'mutual_info': mi_series[feature]
                })

            self.predictive_power_df = (
                pd.DataFrame(records)
                .sort_values(by='pps', ascending=False)
            )
            self.predictive_power_df.to_csv(output_path, index=False)
            print(f"Computed and saved predictive_power df to {output_path}")
        print("="*25)

    def drop_nonstationary_x(self):
        kpss_nonstat = set(self.stationary_df[self.stationary_df['kpss_stationary'] == False].feature)
        # adf_nonstat = set(self.stationary_df[self.stationary_df['adf_stationary'] == False].feature)

        to_drop = kpss_nonstat #& adf_nonstat

        if not to_drop:
            print("No features meet both non-stationary and zero predictive-power criteria.")
            print("="*25)
            return []
        
        self.X_train = self.X_train.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} columns based on stationarity: {sorted(to_drop)}")
        print("="*25)
        return to_drop
    
    def drop_highly_correlated_features(self, threshold, to_drop):
        upper = self.corr_matrix.where(np.triu(np.ones(self.corr_matrix.shape), k=1).astype(bool))
        to_drop2 = set()

        for curr in upper.columns:
            if curr in to_drop or curr in to_drop2:
                continue
            for other in upper.index[upper[curr] > threshold]:
                if other in to_drop or curr in to_drop2:
                    continue

                p_curr_series = self.predictive_power_df.loc[self.predictive_power_df["feature"] == curr, "pps"]
                p_other_series = self.predictive_power_df.loc[self.predictive_power_df["feature"] == other, "pps"]
                p_curr = p_curr_series.iloc[0]
                p_other = p_other_series.iloc[0]

                if p_other > p_curr or (p_other == p_curr and self.corr_matrix[curr].mean() > self.corr_matrix[other].mean()):
                    to_drop2.add(curr)
                else:
                    to_drop2.add(other)

        self.X_train = self.X_train.drop(columns=to_drop2)
        print(f"Dropped {len(to_drop2)} columns based on PPS and corr: {sorted(to_drop2)}")
        print("="*25)
        return to_drop2
    
    def create_pca_features(self, feature_names, n_components=None, variance_threshold=0.9):
        X_cols = []
        for col in feature_names:
            if col in self.features:
                X_cols.append(self.features[col])
            elif col in self.df:
                X_cols.append(self.df[col])
            else:
                raise ValueError(f"Feature '{col}' not found in self.features or self.df")

        X = pd.concat(X_cols, axis=1)
        X.columns = feature_names  

        scaler_df = StandardScaler()
        X_train_full = scaler_df.fit_transform(X.loc[self.X_train.index].copy())
        X_eval_full = scaler_df.transform(X.loc[self.X_eval.index].copy())

        pca_full = PCA()
        pca_full.fit(X_train_full)

        if n_components is None:
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cum_var >= variance_threshold) + 1
            print(f"Auto-selected n_components = {n_components} to retain {variance_threshold*100:.0f}% variance")

        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train_full)
        X_eval_pca = self.pca.transform(X_eval_full)

        pca_cols = [f'pca_{i+1}' for i in range(n_components)]
        X_train_pca_df = pd.DataFrame(X_train_pca, index=self.X_train.index, columns=pca_cols)
        X_eval_pca_df  = pd.DataFrame(X_eval_pca,  index=self.X_eval.index,  columns=pca_cols)

        self.X_train = pd.concat([self.X_train, X_train_pca_df], axis=1)
        self.X_eval  = pd.concat([self.X_eval,  X_eval_pca_df],  axis=1)

        print(f"PCA added {n_components} components to both X_train and X_eval.")
        print("="*25)
        return n_components

    def _ensure_pred_dfs(self):
        if getattr(self, 'train_prediction_df', None) is None:
            self.train_prediction_df = pd.DataFrame(index=self.X_train.index)
        if getattr(self, 'eval_prediction_df', None) is None:
            self.eval_prediction_df = pd.DataFrame(index=self.X_eval.index)

    def train_lightgbm(self, filename, X_features, mask=None, param_grid=None, random_state=42, verbose=False):
        path = f'inputs/{filename}_lgbm.joblib'
        train_prediction_df_path = 'outputs/train_prediction_df.csv'
        eval_prediction_df_path = 'outputs/eval_prediction_df.csv'
        if os.path.exists(train_prediction_df_path) and os.path.exists(eval_prediction_df_path) and os.path.exists(path):
            model = joblib.load(path)
            self.train_prediction_df = pd.read_csv(train_prediction_df_path, index_col=0)
            self.eval_prediction_df = pd.read_csv(eval_prediction_df_path, index_col=0)
            self.train_prediction_df.index = pd.to_datetime(self.train_prediction_df.index)
            self.eval_prediction_df.index = pd.to_datetime(self.eval_prediction_df.index)

            X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
            y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
            X_ev = self.X_eval.loc[mask, X_features]  if mask is not None else self.X_eval[X_features]
            y_ev = self.y_eval.loc[mask]  if mask is not None else self.y_eval
            corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_lgbm'], y_tr)[0] if np.var(y_tr) > 1e-12 else 0.0
            corr_eval = pearsonr(self.eval_prediction_df.loc[X_ev.index, 'pred_lgbm'], y_ev)[0] if np.var(y_ev) > 1e-12 else 0.0
            print(f"lgb train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
            print("="*25)
            return model
        
        grid = param_grid or {
            'num_leaves': [31, 63],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 200],
            'reg_alpha': [0.0, 1.0],
            'reg_lambda': [0.0, 1.0]
        }
        X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
        y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
        X_ev = self.X_eval.loc[mask, X_features] if mask is not None else self.X_eval[X_features]
        y_ev = self.y_eval.loc[mask] if mask is not None else self.y_eval

        best = {'corr': -np.inf, 'params': None, 'model': None}
        for p in ParameterGrid(grid):
            mdl = lgb.LGBMRegressor(**p, objective='regression', random_state=random_state, n_jobs=-1)
            mdl.fit(X_tr, y_tr)
            pred_eval = mdl.predict(X_ev)
            corr = pearsonr(pred_eval, y_ev)[0] if np.var(pred_eval) > 1e-9 else 0.0
            if corr > best['corr']:
                best.update({'corr': corr, 'params': p, 'model': mdl})

        print("Best LGB params:", best['params'], "Eval corr:", best['corr'])
        model = best['model']
        joblib.dump(model, path)

        self._ensure_pred_dfs()
        self.train_prediction_df.loc[X_tr.index, 'pred_lgbm'] = model.predict(X_tr)
        self.eval_prediction_df.loc[X_ev.index, 'pred_lgbm'] = model.predict(X_ev)

        corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_lgbm'], y_tr)[0]
        corr_eval = best['corr']
        print(f"LGBM train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
        print("="*25)
        return model
    
    def train_linear(self, filename, X_features, mask=None, verbose=False):
        path = f'inputs/{filename}_linear.joblib'
        train_prediction_df_path = 'outputs/train_prediction_df.csv'
        eval_prediction_df_path = 'outputs/eval_prediction_df.csv'
        if os.path.exists(train_prediction_df_path) and os.path.exists(eval_prediction_df_path) and os.path.exists(path):
            model = joblib.load(path)
            self.train_prediction_df = pd.read_csv(train_prediction_df_path, index_col=0)
            self.eval_prediction_df = pd.read_csv(eval_prediction_df_path, index_col=0)
            self.train_prediction_df.index = pd.to_datetime(self.train_prediction_df.index)
            self.eval_prediction_df.index = pd.to_datetime(self.eval_prediction_df.index)

            X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
            y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
            X_ev = self.X_eval.loc[mask, X_features]  if mask is not None else self.X_eval[X_features]
            y_ev = self.y_eval.loc[mask]  if mask is not None else self.y_eval
            corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_linear'], y_tr)[0] if np.var(y_tr) > 1e-12 else 0.0
            corr_eval = pearsonr(self.eval_prediction_df.loc[X_ev.index, 'pred_linear'], y_ev)[0] if np.var(y_ev) > 1e-12 else 0.0
            print(f"Linear train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
            print("="*25)
            return model
        
        X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
        y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
        X_ev = self.X_eval.loc[mask, X_features] if mask is not None else self.X_eval[X_features]
        y_ev = self.y_eval.loc[mask] if mask is not None else self.y_eval

        model = LinearRegression().fit(X_tr, y_tr)
        joblib.dump(model, path)
        self._ensure_pred_dfs()
        self.train_prediction_df.loc[X_tr.index, 'pred_linear'] = model.predict(X_tr)
        self.eval_prediction_df.loc[X_ev.index, 'pred_linear'] = model.predict(X_ev)

        corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_linear'], y_tr)[0]
        corr_eval = pearsonr(self.eval_prediction_df['pred_linear'], y_ev)[0]
        print(f"Linear train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
        print("="*25)
        return model

    def train_ridge(self, filename, X_features, mask=None, alphas=[0.01, 0.1, 1.0, 10.0, 100.0], verbose=False):
        path = f'inputs/{filename}_ridge.joblib'
        train_prediction_df_path = 'outputs/train_prediction_df.csv'
        eval_prediction_df_path = 'outputs/eval_prediction_df.csv'
        if os.path.exists(train_prediction_df_path) and os.path.exists(eval_prediction_df_path) and os.path.exists(path):
            model = joblib.load(path)
            self.train_prediction_df = pd.read_csv(train_prediction_df_path, index_col=0)
            self.eval_prediction_df = pd.read_csv(eval_prediction_df_path, index_col=0)
            self.train_prediction_df.index = pd.to_datetime(self.train_prediction_df.index)
            self.eval_prediction_df.index = pd.to_datetime(self.eval_prediction_df.index)

            X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
            y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
            X_ev = self.X_eval.loc[mask, X_features]  if mask is not None else self.X_eval[X_features]
            y_ev = self.y_eval.loc[mask]  if mask is not None else self.y_eval
            corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_ridge'], y_tr)[0] if np.var(y_tr) > 1e-12 else 0.0
            corr_eval = pearsonr(self.eval_prediction_df.loc[X_ev.index, 'pred_ridge'], y_ev)[0] if np.var(y_ev) > 1e-12 else 0.0
            print(f"Ridge train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
            print("="*25)
            return model
        
        X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
        y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
        X_ev = self.X_eval.loc[mask, X_features] if mask is not None else self.X_eval[X_features]
        y_ev = self.y_eval.loc[mask] if mask is not None else self.y_eval

        best = {'corr': -np.inf, 'alpha': None, 'model': None}
        for α in alphas:
            mdl = Ridge(alpha=α, random_state=42).fit(X_tr, y_tr)
            corr = pearsonr(mdl.predict(X_ev), y_ev)[0] if np.var(y_ev) > 1e-9 else 0.0
            if corr > best['corr']:
                best.update({'corr': corr, 'alpha': α, 'model': mdl})
        model = best['model']
        joblib.dump(model, path)
        self._ensure_pred_dfs()
        self.train_prediction_df.loc[X_tr.index, 'pred_ridge'] = model.predict(X_tr)
        self.eval_prediction_df.loc[X_ev.index, 'pred_ridge'] = model.predict(X_ev)

        corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_ridge'], y_tr)[0]
        print(f"Ridge α={best['alpha']} train corr={corr_train:.4f}, eval corr={best['corr']:.4f}")
        print("="*25)
        return model

    def train_xgboost(self, filename, X_features, mask=None, param_grid=None, random_state=42, verbose=False):
        path = f'inputs/{filename}_xgb.joblib'
        train_prediction_df_path = 'outputs/train_prediction_df.csv'
        eval_prediction_df_path  = 'outputs/eval_prediction_df.csv'

        if os.path.exists(train_prediction_df_path) and os.path.exists(eval_prediction_df_path) and os.path.exists(path):
            model = joblib.load(path)
            self.train_prediction_df = pd.read_csv(train_prediction_df_path, index_col=0)
            self.eval_prediction_df = pd.read_csv(eval_prediction_df_path, index_col=0)
            self.train_prediction_df.index = pd.to_datetime(self.train_prediction_df.index)
            self.eval_prediction_df.index = pd.to_datetime(self.eval_prediction_df.index)

            X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
            y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
            X_ev = self.X_eval.loc[mask, X_features]  if mask is not None else self.X_eval[X_features]
            y_ev = self.y_eval.loc[mask]  if mask is not None else self.y_eval
            corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_xgb'], y_tr)[0] if np.var(y_tr) > 1e-12 else 0.0
            corr_eval = pearsonr(self.eval_prediction_df.loc[X_ev.index, 'pred_xgb'], y_ev)[0] if np.var(y_ev) > 1e-12 else 0.0
            print(f"XGBoost train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
            print("="*25)
            return model
        
        grid = param_grid or {
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [200, 400],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0.0, 1.0],
            'reg_lambda': [0.0, 1.0]
        }

        X_tr = self.X_train.loc[mask, X_features] if mask is not None else self.X_train[X_features]
        y_tr = self.y_train.loc[mask] if mask is not None else self.y_train
        X_ev = self.X_eval.loc[mask, X_features]  if mask is not None else self.X_eval[X_features]
        y_ev = self.y_eval.loc[mask]  if mask is not None else self.y_eval

        best = {'corr': -np.inf, 'params': None, 'model': None}
        for p in ParameterGrid(grid):
            mdl = xgb.XGBRegressor(objective='reg:squarederror',
                                random_state=random_state, n_jobs=-1, **p)
            mdl.fit(X_tr, y_tr, eval_set=[(X_ev, y_ev)], verbose=verbose)
            pred_ev = mdl.predict(X_ev)
            corr = pearsonr(pred_ev, y_ev)[0] if (np.var(pred_ev) > 1e-12 and np.var(y_ev) > 1e-12) else 0.0
            if corr > best['corr']:
                best.update({'corr': corr, 'params': p, 'model': mdl})

        print("Best XGB params:", best['params'], "Eval corr:", best['corr'])
        model = best['model']
        os.makedirs('inputs', exist_ok=True)
        joblib.dump(model, path)

        self._ensure_pred_dfs()
        self.train_prediction_df.loc[X_tr.index, 'pred_xgb'] = model.predict(X_tr)
        self.eval_prediction_df.loc[X_ev.index,  'pred_xgb'] = model.predict(X_ev)

        corr_train = pearsonr(self.train_prediction_df.loc[X_tr.index, 'pred_xgb'], y_tr)[0] if np.var(y_tr) > 1e-12 else 0.0
        print(f"XGBoost train corr={corr_train:.4f}, eval corr={best['corr']:.4f}")
        print("="*25)
        return model

    def combine_models(self):
        path = f'inputs/combined_model.joblib'
        train_prediction_df_path = 'outputs/train_prediction_df.csv'
        eval_prediction_df_path = 'outputs/eval_prediction_df.csv'
        if os.path.exists(train_prediction_df_path) and os.path.exists(eval_prediction_df_path) and os.path.exists(path):
            model = joblib.load(path)
            self.train_prediction_df = pd.read_csv(train_prediction_df_path, index_col=0)
            self.eval_prediction_df = pd.read_csv(eval_prediction_df_path, index_col=0)

            corr_train = pearsonr(self.train_prediction_df['pred_final'], self.y_train)[0]
            corr_eval = pearsonr(self.eval_prediction_df['pred_final'], self.y_eval)[0]
            print(f"Final train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
            print("="*25)
            return model
        
        model = LinearRegression().fit(self.train_prediction_df[['pred_linear', 'pred_ridge', 'pred_lgbm', 'pred_xgb']], self.y_train)
        joblib.dump(model, path)
        self._ensure_pred_dfs()
        self.train_prediction_df['pred_final'] = model.predict(self.train_prediction_df[['pred_linear', 'pred_ridge', 'pred_lgbm', 'pred_xgb']])
        self.eval_prediction_df['pred_final'] = model.predict(self.eval_prediction_df[['pred_linear', 'pred_ridge', 'pred_lgbm', 'pred_xgb']])

        corr_train = pearsonr(self.train_prediction_df['pred_final'], self.y_train)[0]
        corr_eval = pearsonr(self.eval_prediction_df['pred_final'], self.y_eval)[0]
        print(f"Final train corr={corr_train:.4f}, eval corr={corr_eval:.4f}")
        print("="*25)
        return model
    
    def save_prediction_df(self):
        self.train_prediction_df.to_csv('outputs/train_prediction_df.csv')
        self.eval_prediction_df.to_csv('outputs/eval_prediction_df.csv')