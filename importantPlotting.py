class Classification_Metrics:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def _plot_confusion_matrix(self, label=None):
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import precision_recall_fscore_support as score
        print(
            '\n\nConfussion Matrix ')
        if label is not None:
            print('\n', confusion_matrix(
                self.y_true, self.y_pred, label))
        else:
            print('\n', confusion_matrix(
                self.y_true, self.y_pred))

        print('\n\nDetails of only positive  class')

        precision, recall, fscore, support = score(
            self.y_true, self.y_pred, average = 'binary')

        print('precision: {}'.format(
            precision))
        print(
            'recall: {}'.format(recall))
        print(
            'fscore: {}'.format(fscore))
        print(
            'support: {}'.format(support))


        print('\n\nDetails of each class')

        precision, recall, fscore, support = score(
            self.y_true, self.y_pred)

        print('precision: {}'.format(
            precision))
        print(
            'recall: {}'.format(recall))
        print(
            'fscore: {}'.format(fscore))
        print(
            'support: {}'.format(support))

    def _plot_roc_curve(self, pos_label=None,predicted_prob=None):
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        false_positive_rate, true_positive_rate, thresholds = roc_curve(
            self.y_true, predicted_prob,pos_label=pos_label)
        roc_auc = auc(
            false_positive_rate, true_positive_rate)
        print(
            '\nROC auc -', roc_auc)
        plt.title(
            'ROC')
        plt.plot(false_positive_rate, true_positive_rate, 'b',
                 label='AUC = %0.2f' % roc_auc)
        plt.legend(
            loc='lower right')
        plt.plot(
            [0, 1], [0, 1], 'r--')
        plt.ylabel(
            'True Positive Rate')
        plt.xlabel(
            'False Positive Rate')
        plt.show()
