  import csv
from sklearn import preprocessing

def get_data_entries() :
	data_entries = []
	results = []
	genres = []
	with open('dataset/track_data.csv', encoding="ISO-8859-1") as f:
		f.readline()
		content = f.readlines()

		for line in content:
			data_entry = line.strip('\n').split(',')
			a,b,c,d,e,f,g,h = data_entry
			genres.append(e)
			data_entries.append([c,d,h,f])
			results.append(g)

	le = preprocessing.LabelEncoder()
	genre_labels = le.fit_transform(genres)
	data_entries = data_entries
	return [data_entries, results]
