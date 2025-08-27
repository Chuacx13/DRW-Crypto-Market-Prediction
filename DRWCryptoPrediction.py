import pandas as pd
import numpy as np
import os
import joblib

class DRWCryptoPrediction:
    def __init__(self, df, pca, standard_scaler):
        self.df = df
        self.features = pd.DataFrame(index=self.df.index)
        self.X_test = None
        self.test_prediction_df = None
        self.pca = pca
        self.standard_scaler = standard_scaler

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

    def standardize_features(self):
        cols = [col for col in self.X_test.columns if 'label' not in col]
        self.X_test[cols] = self.standard_scaler.transform(self.X_test[cols])

        print("Standardizing features")
        print("="*25)

    def create_X_test_df(self):
        self.X_test = pd.concat([self.df, self.features], axis=1)
        print("Created X_test dataframe")
        print("="*25)

    def check_standardised_features(self):
        means = self.X_test.mean()
        stds = self.X_test.std()

        print("Mean across X-features (abs avg):", means.abs().mean())
        print("Std across X-features (avg):", stds.mean())
        print("="*25)

    def drop_features(self, columns):
        self.X_test = self.X_test.drop(columns=list(columns))
    
    def create_pca_features(self, feature_names, n_components):
        X_cols = []
        for col in feature_names:
            if col in self.X_test:
                X_cols.append(self.X_test[col])
            else:
                raise ValueError(f"Feature '{col}' not found in self.X_test")

        X_test_full = pd.concat(X_cols, axis=1)
        X_test_full.columns = feature_names  

        X_test_pca = self.pca.transform(X_test_full)

        pca_cols = [f'pca_{i+1}' for i in range(n_components)]
        X_test_pca_df = pd.DataFrame(X_test_pca, index=self.X_test.index, columns=pca_cols)

        self.X_test = pd.concat([self.X_test, X_test_pca_df], axis=1)

        print(f"PCA added {n_components} components to X_test.")
        print("="*25)

    def _ensure_pred_dfs(self):
        if getattr(self, 'test_prediction_df', None) is None:
            self.test_prediction_df = pd.DataFrame(index=self.X_test.index)

    def predict(self, filename, X_features, random_state=42, verbose=False):
        lgb_path = f'inputs/{filename}_lgbm.joblib'
        xgb_path = f'inputs/{filename}_xgb.joblib'
        linear_path = f'inputs/{filename}_linear.joblib'
        ridge_path = f'inputs/{filename}_ridge.joblib'
        combined_path = f'inputs/combined_model.joblib'

        test_prediction_df_path = 'outputs/test_prediction_df.csv'
        if os.path.exists(test_prediction_df_path):
            self.test_prediction_df = pd.read_csv(test_prediction_df_path, index_col=0)
            print("Loaded test predictions")
            print("="*25)
        
        X = self.X_test[X_features]

        lgb_model = joblib.load(lgb_path)
        xgb_model = joblib.load(xgb_path)
        linear_model = joblib.load(linear_path)
        ridge_model = joblib.load(ridge_path)
        combined_model = joblib.load(combined_path)

        self._ensure_pred_dfs()
        self.test_prediction_df['pred_lgbm']   = lgb_model.predict(X)
        self.test_prediction_df['pred_xgb']    = xgb_model.predict(X)
        self.test_prediction_df['pred_linear'] = linear_model.predict(X)
        self.test_prediction_df['pred_ridge']  = ridge_model.predict(X)
        self.test_prediction_df['pred_final'] = combined_model.predict(self.test_prediction_df[['pred_linear', 'pred_ridge', 'pred_lgbm', 'pred_xgb']])

        print(f"Completed predictions")
        print("="*25)
    
    def save_prediction_df(self):
        self.test_prediction_df.to_csv('outputs/test_prediction_df.csv')

    def create_submission_file(self, filename):
        out = (self.test_prediction_df[['pred_final']]
            .rename(columns={'pred_final': 'prediction'})
            .reset_index()
            .rename(columns={'index': 'ID'}))
        out.to_csv(f'outputs/submission_{filename}.csv', index=False)