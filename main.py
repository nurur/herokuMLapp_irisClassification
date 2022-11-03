
import numpy as np
import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

import streamlit as st

st.set_page_config(layout = "wide")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)
pd.set_option('display.max_colwidth', None)


######################################################################### 
# Load Data
@st.cache(allow_output_mutation=False)
def get_data() :
    from sklearn.datasets import load_iris
    df = load_iris(as_frame=True)
    X  = df.data
    y  = df.target

    oldColNames = df.feature_names 
    newColNames = [x.split('(cm)')[0].strip().title() for x in oldColNames]
    nameDict = {oldColNames[i] : newColNames[i] for i in range(len(oldColNames))}
    X = X.rename( columns=nameDict, inplace=False)
    Y = pd.DataFrame( y.rename('Labels') )

    data = pd.concat([X,Y], axis=1)

    unique_labels  = data.Labels.unique().tolist() 
    unique_species = df.target_names
    mapper_species = { unique_labels[x] : unique_species[x] for x in range(len(unique_labels)) }
    data['Species'] = data['Labels'].apply( lambda  row : mapper_species[row] )
    
    return data

# data = pydata.data('iris')
# oldColNames = data.columns.tolist()
# newColNames = [ re.sub("\."," ",x) for x in data.columns.tolist() ]

# nameDict = {oldColNames[i] : newColNames[i] for i in range(len(oldColNames))}
# data = data.rename( columns=nameDict )

data = get_data()
data = data.drop(columns=['Labels'])

sl_min = data['Sepal Length'].min()
sl_max = data['Sepal Length'].max()
sw_min = data['Sepal Width'].min()
sw_max = data['Sepal Width'].max()
pl_min = data['Petal Length'].min()
pl_max = data['Petal Length'].max()
pw_min = data['Petal Width'].min()
pw_max = data['Petal Width'].max()


######################################################################### 
# Utility Function for the Sidebar
def user_input_features():
	sepal_length = st.sidebar.slider("Sepal Length", min_value=float(sl_min), max_value=float(sl_max) )
	sepal_width  = st.sidebar.slider("Sepal Width",  min_value=float(sw_min), max_value=float(sw_max) )
	petal_length = st.sidebar.slider("Petal Length", min_value=float(pl_min), max_value=float(pl_max) )
	petal_width  = st.sidebar.slider("Petal Width",  min_value=float(pw_min), max_value=float(pw_max) )

	dataDict = {"sepal length" : float(sepal_length), 
				"sepal width"  : float(sepal_width), 
				"petal length" : float(petal_length), 
				"petal width"  : float(petal_width)}

	features = pd.DataFrame(dataDict, index=[0])
	features['newIndex'] = ''
	features = features.set_index('newIndex')
	features.index.name = ''
	return features

######################################################################### 
# Build Sidebar
st.sidebar.header("Select Input Parameters")
df_user_input = user_input_features()


# st.subheader("Showing Selected Parameters")
# #col1, col2, col3, col4 = st.columns([1,2,3,4])
# #with col1:
# st.table( df_user_input )


st.sidebar.header("Select Model")
select_model = st.sidebar.selectbox(
	'Select a Modeling Technique',
	("Click to Select", "Logistic Regression", "K Nearest Neighbors", "Random Forest"))

######################################################################### 
# Build the Main Page
# st.write("""
# 	# Interactive Classification App 
# 	### Using Python, Plotly, and Streamlit
# 	#### Created by Nurur Rahman
# 	""")
mkdtext = """
		# Prediction App 
		### IRIS Species Type
		<font color='red'> Created by Nurur Rahman </font>
		"""
st.markdown(mkdtext, unsafe_allow_html=True)
st.write(" ")


st.info(
	"""
	App Properties : \n
	1. Classification using IRIS dataset.
	2. Multiple ML techniques applied to a single dataset 
	3. The data is selected manually from the sidebar.
	"""
)
st.write("  ")

######################################################################### 
st.warning("Data Table")
showButton = st.button("Click to View the Table")
if showButton :
	#st.subheader("Showing the Data")
	st.write( data )

	hideCheckBox = st.checkbox("Click to Hide the Table")
	if hideCheckBox:
		showButton = False

st.write("  ")

st.info("Class Labels and the Corresponding Indices")
# col1, col2, col3, col4 = st.columns([1,2,2,2])
# with col1:
# 	classLabels  = data.Species.unique()
# 	classIndices = np.array([0,1,2])
	
# 	labels_indices = pd.DataFrame( np.column_stack( (classLabels,classIndices) ), 
# 		columns=['Lables','Indices'])
# 	labels_indices['newIndex'] = ''
# 	labels_indices = labels_indices.set_index('newIndex') 
# 	labels_indices.index.name = ''
# 	st.table(  labels_indices )

classLabels  = data.Species.unique()
classIndices = np.array([0,1,2])

labels_indices = pd.DataFrame( np.column_stack( (classLabels,classIndices) ), 
columns=['Lables','Indices'])
labels_indices['newIndex'] = ''
labels_indices = labels_indices.set_index('newIndex') 
labels_indices.index.name = ''
#st.table( labels_indices )
st.write( labels_indices )
st.write(" ")


st.info("User Input Data")
#col1, col2, col3, col4 = st.columns([1,2,3,4])
#with col1:
# st.table( df_user_input )

showButton = st.button("Click to Show the Data",  key='user_input')
if showButton :
	st.write( df_user_input )
	hideCheckBox = st.checkbox("Click to Hide the Data")
	if hideCheckBox:
		showButton = False

st.write(" ")
######################################################################### 
# Build the Model
X = data.drop(columns=['Species'], axis=0)
y = data['Species']


if select_model=='Click to Select':
	#clf = 'to be selected'
	pass
elif select_model=='Logistic Regression':
	clf = LogisticRegression()
	clf.fit(X,y)
elif select_model=='K Nearest Neighbors':
	clf = KNeighborsClassifier(n_neighbors=3)
	clf.fit(X, y)
else:
	clf = RandomForestClassifier()
	clf.fit(X,y)


if select_model != 'Click to Select':
	try:
		st.success(f"User Selected ML Model : {select_model}")

		pred_proba = clf.predict_proba(df_user_input)
		#pred_proba = np.array( [ round(x,2) for x in pred_proba[0] ] )
		pred_proba = [ round(x,2) for x in pred_proba[0] ]
		pred_class = clf.predict(df_user_input)

		st.write(" ")

		st.info("Predicted Probability for the Input Features")
		# col1, col2, col3, col4 = st.columns([1,2,2,2])
		# with col1:
		# 	df_proba1 = pd.DataFrame( np.array([pred_proba]), columns=classLabels)
		# 	df_proba1['newIndex'] = ''
		# 	df_proba2 = df_proba1.set_index('newIndex')
		# 	df_proba2.index.name = ''
		# 	st.table( df_proba2 )
		df_proba1 = pd.DataFrame( np.array([pred_proba]), columns=classLabels )
		df_proba1['newIndex'] = ''
		df_proba2 = df_proba1.set_index('newIndex')
		df_proba2.index.name = ''
		#st.table( df_proba2 )
		st.write( df_proba2 )


		st.info("Predicted Class for the Input Features")
		col1, col2 = st.columns([1,2])
		with col1:
			#st.write( pred_class )
			df_class1 = pd.DataFrame( np.array([pred_class]), columns=['Label'] )
			df_class1['newIndex'] = ''
			df_class2 = df_class1.set_index('newIndex')
			df_class2.index.name = ''
			#st.table( df_class2 )
			st.write( df_class2 )

	
	except NameError:
		st.info("Please Select a Model from the Sidebar")







