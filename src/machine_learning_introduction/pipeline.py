import typing
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Pipeline:
    def __init__(self, 
                 scaler: typing.Optional[TransformerMixin], 
                 fill_categorical_strategy: typing.Callable[[pd.Series], any], 
                 fill_numerical_strategy: typing.Callable[[pd.Series], float],
                 exclude_collumns: typing.List[str],
                 model: BaseEstimator,
                 y_column: str,
                 identifier: str,
                 test_size: float = 0.1):
        self.scaler = scaler
        self.fill_categorical_strategy = fill_categorical_strategy
        self.fill_numerical_strategy = fill_numerical_strategy
        self.exclude_collumns = exclude_collumns
        self.model = model
        self.y_column = y_column
        self.identifier = identifier
        self.test_size = test_size
    
    def fill_categorical(self, df: pd.DataFrame):
        """Fills categorical values using the provided strategy."""
        for col in df.select_dtypes(include=['object']).columns:
            if col != self.identifier and col != self.y_column:
                df[col] = df[col].fillna(self.fill_categorical_strategy(df[col]))
        return df

    def encode_categorical(self, df: pd.DataFrame):
        """Converts categorical variables to dummies."""
        # Identify categorical columns excluding identifiers/targets
        cols_to_encode = [col for col in df.select_dtypes(include=['object']).columns 
                          if col != self.identifier and col != self.y_column]
        
        if not cols_to_encode:
            return df

        df_dummies = pd.get_dummies(df[cols_to_encode], drop_first=True)
        df = df.drop(columns=cols_to_encode).join(df_dummies)
        return df
    
    def fill_numerical(self, df: pd.DataFrame):
        """Fills numerical values using the provided strategy."""
        # Select numerical columns excluding target/identifier
        cols = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in cols if c != self.identifier and c != self.y_column]

        for col in cols:
            df[col] = df[col].fillna(self.fill_numerical_strategy(df[col]))
        return df
    
    def normalize(self, df: pd.DataFrame):
        """Normalizes numerical data if a scaler is provided."""
        if self.scaler is None:
            return df
            
        numerical_cols = df.select_dtypes(include=[np.number, int, float, bool]).columns
        # Filter out identifiers and targets from scaling
        cols_to_scale = [c for c in numerical_cols if c != self.identifier and c != self.y_column]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
            
        return df

    def view_config(self):
        print(f"Model: {self.model}")
        print(f"Scaler: {self.scaler}")
        print(f"Cat Strategy: {self.fill_categorical_strategy.__name__}")
        print(f"Num Strategy: {self.fill_numerical_strategy.__name__}")
        print(f"Excluded: {self.exclude_collumns}")

    def process(self, train: pd.DataFrame, validate: pd.DataFrame, see_action: bool = False):
        """Process training and validation data, fit the model, and predict."""
        
        # Save target and identifier
        y_train = train[self.y_column]
        ids_validate = validate[self.identifier] if self.identifier in validate.columns else None

        # Drop excluded columns
        if self.exclude_collumns:
            train = train.drop(columns=[c for c in self.exclude_collumns if c in train.columns])
            validate = validate.drop(columns=[c for c in self.exclude_collumns if c in validate.columns])
        
        # Preprocessing Pipeline
        train = self.fill_categorical(train)
        validate = self.fill_categorical(validate)
        if see_action: print("Categorical data filled")
        
        train = self.encode_categorical(train)
        validate = self.encode_categorical(validate)
        if see_action: print("Data encoded")
        
        train = self.fill_numerical(train)
        validate = self.fill_numerical(validate)
        if see_action: print("Numerical data filled")
        
        # Align columns (handle missing dummy columns in validation)
        train, validate = train.align(validate, join="left", axis=1, fill_value=0)
        
        # Restore target and identifier after alignment (alignment might mess them up if missing)
        train[self.y_column] = y_train
        if ids_validate is not None:
            validate[self.identifier] = ids_validate

        if see_action: print("Columns aligned")
        
        train = self.normalize(train)
        validate = self.normalize(validate)
        if see_action: print("Data normalized")
        
        # Prepare X and y
        X_train = train.drop(columns=[self.y_column, self.identifier], errors='ignore')
        y_train = train[self.y_column]
        
        X_validate = validate.drop(columns=[self.y_column, self.identifier], errors='ignore')

        # Fit and Predict
        self.model.fit(X_train, y_train)
        prediction = self.model.predict(X_validate)
        
        # Format answer
        answer = pd.DataFrame()
        if ids_validate is not None:
            answer[self.identifier] = ids_validate
        answer[self.y_column] = prediction
        
        return train, answer
