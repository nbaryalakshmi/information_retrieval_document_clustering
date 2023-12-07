import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input,Output,State
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download("stopwords")
nltk.download("punkt")
stopwords = stopwords.words('english')
def clean_text(text):
    port_stemmer = PorterStemmer()
    lst_tokens = word_tokenize(text)
    text_cleaned = ""
    for word in lst_tokens:
        if word not in stopwords:
            text_cleaned += port_stemmer.stem(word) + " "
    return text_cleaned

def get_cluster(input_text):
    with open('clustering_model.pickle', 'rb') as f:
        kmeans_model = pickle.load(f)
    with open('vectorizer.pickle', 'rb') as f:
        vectorizer = pickle.load(f)
    input_text=clean_text(input_text)
    input_vectorized = vectorizer.transform([input_text])
    predicted_label = kmeans_model.predict(input_vectorized)
    return predicted_label

app = dash.Dash(__name__,meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],
                suppress_callback_exceptions=True)
#Define the layout of the app
app.layout = dbc.Container(
    children=[
        html.Div([dbc.Label('Please enter text to cluster.')],className="col-12"),
        html.Div([dbc.Input(id='input_text', type='text', value='')],className="col-12"),
        html.Div(" "),
         html.Div(
    [
        dbc.Button('Submit', id='btn_search',outline=True, color="success",size="lg",)
    ],
    className="col-12 mx-auto",
                    ),
        html.Div(id='ul_results',children=['Please enter text to cluster.'],className="col-12"),
                
    ]
)
#Get result set
@app.callback(
    dash.dependencies.Output('ul_results', 'children'),
    [Input('btn_search', 'n_clicks')],
    [State('input_text', 'value'),
     State('ul_results', 'children')]
)
def get_result_set(n_clicks, input_text, current_list):
    if current_list==None:
        current_list=[]
    if n_clicks is None:
        return current_list
    else:
        if input_text.strip()=="":
            return ["Please enter text to cluster."]
        else:
            current_list=[]
            res_cluster=get_cluster(input_text)
            #2:Health,3:climate,1:technology,0:sports
            if res_cluster[0]==2:                
                current_list.append("Predicted cluster of given text: Health")
            elif res_cluster[0]==3:                
                current_list.append("Predicted cluster of given text: Climate")
            elif res_cluster[0]==1:                
                current_list.append("Predicted cluster of given text: Technology")
            elif res_cluster[0]==0:                
                current_list.append("Predicted cluster of given text: Sports")
            return current_list 
if __name__ == '__main__':
    app.run_server(debug=True)