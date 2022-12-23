from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class BuildPipeline:

    """
    This class builds a pipeline of transformers and model. Also fits grid search
    given input data. Can return parameters, predictions and score.
    IT DOES NOT IMPLEMENT `fit` METHOD OF ANY CLASSIFIER. However, since 
    `get_pipeline()` method returns a pipeline object (which presumably implements)
    `fit` method, a model can be fit then.
    """
    
    def __init__(self, X, y):

        """
        Parameters :
        -------------------
        X : pandas DataFrame.
            Features to train a model on.

        y : pandas Series / numpy array.
            Target feature to train a model on.
        """

        self.X = X
        self.y = y


    def get_pipeline(self, transformer, model, cat_cols=[], num_cols=[], **kwargs):
        
        """
        Transform input dataframe / Series X using `transformer` and pass
        it into a pipeline where X and y are used to train `model` passed
        into when class instance was defined.
        A pipeline object is defined (but not fitted).

        Parameters : 
        -------------------
        transformer : str.
            The transformer to transform X. Possible options:
            - 'ct': ColumnTransformer.
                    One-hot encode the categorical columns `cat_cols` of X, 
                    standardize and generate polynomial and interaction terms 
                    for the numerical columns `num_cols`.
            - 'cvec': CountVectorizer.
                    Use CountVectorizer in a pipeline. Only relevant for an 
                    input of Series of string literals.
            - 'tvec': TfidfVectorizer.
                    Use TfidfVectorizer in a pipeline. Only relevant for an 
                    input of Series of string literals.

        cat_cols : list of strings, default: [].
            The list of categorical columns of X that are to be one-hot encoded.

        num_cols : list of strings, default: [].
            The list of numerical columns of X that are to be transformed.

        model : scikit-learn regression / classifier type object
            The model to be trained on (X, y). Must be a Python class object that 
            implements `fit()` method.

        Returns :
        -------------------
        None.
        """

        if transformer == 'ct':
            tr = ColumnTransformer([
                ('oh', OneHotEncoder(drop='first'), cat_cols),
                ('sc', StandardScaler(), num_cols),
                ('poly', PolynomialFeatures(include_bias=False), num_cols),
                ], remainder='passthrough')
        elif transformer == 'cvec':
            tr = CountVectorizer()
        elif transformer == 'tvec':
            tr = TfidfVectorizer()
        else:
            raise ValueError(f"{transformer} is not a valid transformer name.")

        name = ''.join([char.lower() for char in model.__name__ if char.isupper()])
        
        self.pipe = Pipeline([
            (transformer, tr),
            (name, model(**kwargs))
        ])



    def get_params(self):
        """
        Get the parameters of the pipeline object. Main use case is to 
	get a full list of the parameters to tune in Grid Search.        
        """
        return self.pipe.get_params()


    def grid_search(self, param_grid={}, cv=1, random=False, **kwargs):
        """
        Fits the pipeline defined in `get_pipeline()` to X and y. If cv > 1, 
	implements scikit-learn's GridSearchCV (or RandomizedSearchCV if 
	random=True) on the pipeline; otherwise simply fits the pipeline.


        Parameters:
        -------------------
	param_grid : dict or list of dicts, default: {}.
	    The parameter grid to search over.

	cv: int, default: 1.
	    Determines cross-validation splitting strategy. If it is equal to 1
	    there will be no grid search (even if param_grid has value passed in)
	    and the pipeline will be fitted to X and y. If any integer greater 
	    than 1, a grid search will be performed.

	random: bool, default: False.
	    Determines whether to implement GridSearchCV or RandomizedSearchCV.
	    If random=True, implement RandomizedSearchCV; implement GridSearchCV 
	    otherwise.

	Returns :
        -------------------
	The best estimator from grid search.
        """

# e.g.
#         param_grid = {
#          'ct__poly__degree': [1, 2],
#          'model_alpha': np.logspace(-4, -1, 10) 
#         }

        if cv <= 1:
            return self.pipe.fit(self.X, self.y)
        else:
            kf = KFold(n_splits=cv, shuffle=True)

            if random:
                return RandomizedSearchCV(self.pipe, param_grid, cv=kf, **kwargs).fit(self.X, self.y)
            else:
                return GridSearchCV(self.pipe, param_grid, cv=kf).fit(self.X, self.y)
