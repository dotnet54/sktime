#python implementation of ts-chief 
#algorithm plan

#---------------------------------

def __main__():

	train_x, train_y, test_x, test_y = load_dataset('Fish')

	params = {
		'C_e' : 5
	}


	model = new ChiefForest(**params)

	model.fit(train_x, train_y)
	predicted_y = model.predict(test_x)	
	score = accuracy_score(test_y, predicted_y)


def ChiefForest extends BaseClassifier,Ensemble:

	def __init__():


	def fit(train_x, train_y):

		return None

	def predict(test_x, test_y):

		return None


def ChiefTree extends BaseClassifier:


	def __init__():


	def fit(train_x, train_y):

		return None

	def predict(test_x, test_y):

		return None

