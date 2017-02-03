import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
from sklearn.cross_validation import cross_val_score, cross_val_predict, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, mean_squared_error
import matplotlib.pyplot as plt
seed = 7
np.random.seed(seed)

class TrainingHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.predictions = []
        self.i = 0
        self.save_every = 50

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.i += 1        
        if self.i % self.save_every == 0:        
            pred = model.predict(x_train)
            self.predictions.append(pred)
            
class Tools:

    def __init__(self, *args):
        pass

    def _grid_search(self, x_train, y_train, model, params, cval=5):
        kfold = KFold(
            n=cval, random_state=seed)
        grid = GridSearchCV(
            estimator=model, param_grid=params, cv=kfold)
        grid_result = grid.fit(
            x_train, y_train)
        print("Best: %f using %s" % (
            grid_result.best_score_, grid_result.best_params_))
        return grid_result.best_params_

    def _cross_validation(self, x_train, y_train, extimator, cval=5, *args):
        kfold = KFold(
            n=cval, random_state=seed)
        results = cross_val_score(
            estimator, x_train, y_train, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (
            results.mean()*100, results.std()*100))


class Multiclass_Without_Grid_Search:

    def __init__(self, x, y, *arg):
        self.x = x
        self.y = y

    def _prepare_test_data(self, test_size=0.2, *arg):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed)

    def _model_build(self, nb_epoch=100, batch_size=10, *arg):
        model = Sequential()
        model.add(Dense(
            4, input_dim=4, init='normal', activation='relu'))
        model.add(Dense(
            3, init='normal', activation='sigmoid'))
        model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self._prepare_test_data(
            test_size=0.2)
        valdation_iter = 5
        test_size_for_validation = 0.4
        idx = StratifiedShuffleSplit(
            self.y_train, n_iter=valdation_iter, test_size=test_size_for_validation, random_state=seed)
        for train_idx, test_idx in idx:
            xx_train, xx_test, yy_train, yy_test = self.x_train[train_idx], self.x_train[
                test_idx], self.y_train[train_idx], self.y_train[test_idx]
            model.fit(
                xx_train, yy_train, nb_epoch=nb_epoch, batch_size=batch_size)
            yy_pred = model.predict(
                xx_test)
            # Convert 3d
            # oupput to
            # 1
            y_pred = np.argmax(
                yy_pred, axis=1)
            y_test = np.argmax(
                yy_test, axis=1)
            print('F1 score - ', f1_score(
                y_test, y_pred, average="macro"))
            print('Precision - ', precision_score(
                y_test, y_pred, average="macro"))
            print('Recall - ', recall_score(
                y_test, y_pred, average="macro"))

        model.fit(
            self.x_train, self.y_train, nb_epoch=nb_epoch, batch_size=batch_size)
        yy_pred = model.predict(
            self.x_test)
        self.y_pred = np.argmax(
            yy_pred, axis=1)
        self.y_true = np.argmax(
            self.y_test, axis=1)
        self.prob = model.predict_proba(
            self.x_test)
        self._analyse_result()

    def _analyse_result(self, *arg):
        from importantPlotting import Classification_Metrics
        obj = Classification_Metrics(
            self.y_true, self.y_pred)
        obj._plot_confusion_matrix()
        no_classes = self.prob.shape[
            1]
        # Plot ROC
        for clas in range(no_classes):
            obj._plot_roc_curve(
                pos_label=clas, predicted_prob=self.prob[:, clas])

        import gc
        gc.collect()


class Binary_Without_Grid_Search(TrainingHistory):

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.history = TrainingHistory()

    def _prepare_test_data(self, test_size=0.2, *arg):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed)

    def _model_build(self, nb_epoch=100, batch_size=10, *arg):
        # create
        # model
        model = Sequential()
        model.add(Dense(8, input_dim=8,
                        init='uniform', activation='relu'))
        model.add(Dense(
            8, init='uniform', activation='relu'))
        model.add(Dense(
            1, init='uniform', activation='sigmoid'))

        # Compile
        # model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        self._prepare_test_data(
            test_size=0.2)
        valdation_iter = 5
        test_size_for_validation = 0.4
        idx = StratifiedShuffleSplit(
            self.y_train, n_iter=valdation_iter, test_size=test_size_for_validation, random_state=seed)
        for train_idx, test_idx in idx:
            xx_train, xx_test, yy_train, yy_test = self.x_train[train_idx], self.x_train[
                test_idx], self.y_train[train_idx], self.y_train[test_idx]
            model.fit(
                xx_train, yy_train, nb_epoch=nb_epoch, batch_size=batch_size)
            yy_pred = model.predict(
                xx_test)
            # Convert 3d
            # oupput to
            # 1
            y_pred = [np.round(
                x) for x in yy_pred]
            y_test = yy_test
            print('F1 score - ', f1_score(
                y_test, y_pred, average="binary"))
            print('Precision - ', precision_score(
                y_test, y_pred, average="binary"))
            print('Recall - ', recall_score(
                y_test, y_pred, average="binary"))

        # Prediction
        # will
        # always be
        # on
        # complete
        # data
        model.fit(
            self.x_train, self.y_train, nb_epoch=nb_epoch, batch_size=batch_size)
        yy_pred = model.predict(
            self.x_test)
        self.y_pred = [np.round(
            x) for x in yy_pred]
        self.y_true = self.y_test
        self.prob = model.predict_proba(
            self.x_test)
        self._analyse_result()

    def _analyse_result(self, *arg):
        from importantPlotting import Classification_Metrics
        obj = Classification_Metrics(
            self.y_true, self.y_pred)
        obj._plot_confusion_matrix()
        obj._plot_roc_curve(
            pos_label=1, predicted_prob=self.prob)
        # plt.figure(figsize=(6, 3))
        # plt.plot(self.history.losses)
        # plt.ylabel('error')
        # plt.xlabel('iteration')
        # plt.title('training error')
        # plt.show()

        import gc
        gc.collect()


class Binary_With_Grid_Search:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def _prepare_test_data(self, test_size=0.2, *arg):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed)

    # Function to
    # create model,
    # required for
    # KerasClassifier
    def create_model(self, optimizer='adam', init='uniform'):
        # create
        # model
        model = Sequential()
        model.add(Dense(
            12, input_dim=8, init=init, activation='relu'))
        model.add(Dense(
            8, init=init, activation='relu'))
        model.add(Dense(
            1, init=init, activation='sigmoid'))
        # Compile
        # model
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer, metrics=['accuracy'])
        return model

    def _model_build(self, *arg):
        self._prepare_test_data()
        model = KerasClassifier(
            build_fn=self.create_model, verbose=0)
        optimizers = [
            'adam']
        init = [
            'normal', 'uniform']
        epochs = [
            100, 150]
        batches = [
            5, 10]
        param_grid = dict(
            optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init)
        grid = GridSearchCV(
            estimator=model, param_grid=param_grid, cv=5)
        grid_result = grid.fit(
            self.x_train, self.y_train)
        print("Best: %f using %s" % (
            grid_result.best_score_, grid_result.best_params_))
        # means = grid_result.cv_results_[
        #     'mean_test_score']
        # stds = grid_result.cv_results_[
        #     'std_test_score']
        # params = grid_result.cv_results_[
        #     'params']
        # for mean, stdev, param in zip(means, stds, params):
        #     print("%f (%f) with: %r" % (
        # mean,
        # stdev,
        # param))

        # Training
        # with Best
        # Parameter
        model = Sequential()
        model.add(Dense(
            12, input_dim=8, init=grid_result.best_params_['init'], activation='relu'))
        model.add(Dense(
            8, init=grid_result.best_params_['init'], activation='relu'))
        model.add(Dense(
            1, init=grid_result.best_params_['init'], activation='sigmoid'))
        model.compile(loss='binary_crossentropy',
                      optimizer=grid_result.best_params_['optimizer'], metrics=['accuracy'])
        # Compile
        # model
        model.fit(
            self.x_train, self.y_train, nb_epoch=grid_result.best_params_['nb_epoch'], batch_size=grid_result.best_params_['batch_size'])
        yy_pred = model.predict(
            self.x_test)
        self.y_pred = [np.round(
            x) for x in yy_pred]
        self.y_true = self.y_test
        self.prob = model.predict_proba(
            self.x_test)
        self._analyse_result()

    def _analyse_result(self, *arg):
        from importantPlotting import Classification_Metrics
        obj = Classification_Metrics(
            self.y_true, self.y_pred)
        obj._plot_confusion_matrix()
        obj._plot_roc_curve(
            pos_label=1, predicted_prob=self.prob)

        import gc
        gc.collect()


class Regression_Without_Grid_Search:

    def __init__(self, x, y, *arg):
        self.x = x
        self.y = y

    def _prepare_test_data(self, test_size=0.2, *arg):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed)

    # define base
    # mode
    def create_model(self):
        # create
        # model
        model = Sequential()
        model.add(Dense(
            13, input_dim=13, init='normal', activation='relu'))
        model.add(
            Dense(1, init='normal'))
        # Compile
        # model
        model.compile(
            loss='mean_squared_error', optimizer='adam')
        return model

    def _model_build(self, *arg):
        self._prepare_test_data()
        estimators = []
        estimators.append(
            ('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.create_model, nb_epoch=50, batch_size=5, verbose=1)))
        pipeline = Pipeline(
            estimators)
        kfold = KFold(
            n=5, random_state=seed)
        results = cross_val_score(
            pipeline, self.x_train, self.y_train, cv=kfold)
        print(
            results, '\n')
        print("Standardized: %.2f (%.2f) MSE" % (
            results.mean(),
            results.std()))

        pipeline.fit(
            self.x_train, self.y_train)
        yy_pred = pipeline.predict(
            self.x_test)
        # yy_pred = cross_val_predict(
        # pipeline,
        # self.x_test,
        # cv=kfold)
        self.y_pred = yy_pred
        self.y_true = self.y_test
        self._analyse_result()

    def _analyse_result(self, *arg):
        print('RMSE on the test data \n', mean_squared_error(
            self.y_true, self.y_pred)**0.5)


class Regression_With_Grid_Search(Tools):

    def __init__(self, x, y, *arg):
        self.x = x
        self.y = y

    def _prepare_test_data(self, test_size=0.2, *arg):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=seed)

    # define base
    # mode
    def create_model(self, init='normal', optimizer='adam'):
        # create
        # model
        model = Sequential()
        model.add(Dense(
            13, input_dim=13, init=init, activation='relu'))
        model.add(
            Dense(1, init=init))
        # Compile
        # model
        model.compile(
            loss='mean_squared_error', optimizer=optimizer)
        return model

    def _model_build(self, *arg):
        self._prepare_test_data()
        estimators = []
        estimators.append(
            ('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.create_model, nb_epoch=50, batch_size=5, verbose=0)))
        pipeline = Pipeline(
            estimators)
        optimizers = [
            'adam']
        init = [
            'normal', 'uniform']
        epochs = [
            100, 150]
        batches = [
            5, 10]
        params = dict(mlp__optimizer=optimizers, mlp__nb_epoch=epochs,
                      mlp__batch_size=batches, mlp__init=init)
        best_params = Tools()._grid_search(
            self.x_train, self.y_train, pipeline, params, 5)

        # Creating
        # best model
        #Change init and optimezer of create model acc to bbest params
        estimators = []
        estimators.append(
            ('standardize', StandardScaler()))
        estimators.append(('mlp', KerasRegressor(
            build_fn=self.create_model, nb_epoch=best_params['mlp__nb_epoch'],
            batch_size=best_params['mlp__batch_size'], verbose=0)))
        pipeline = Pipeline(
            estimators)
        pipeline.fit(
            self.x_train, self.y_train)
        self.y_pred = pipeline.predict(
            self.x_test)
        self.y_true = self.y_test
        self._analyse_result()

    def _analyse_result(self, *args):
        print('RMSE on the test data \n', mean_squared_error(
            self.y_true, self.y_pred)**0.5)
