class EDSplitter:
    params = None
    measure = None
    exemplars = []
    tree = None

    def __init__(self, tree, **params):
        self.tree = tree
        self.params = params

    def init_data(self):
        print('init data')

    def split(self, X_train, y_train):

        #select a random measure
        measure = self.euclidean

        #select a random param

        #select random exemplars
        cls_indices = self.tree.class_indices


        #partition based on similarity to the exemplars

        return []

    def euclidean(self, a, b):

        return 0;

    def dtw(self, a, b):

        return 0;