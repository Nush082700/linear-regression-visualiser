import os
from flask import Flask, request, render_template, url_for, redirect
import model as mdl
import plotly
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.io as pio
import json


app = Flask(__name__)


def upload_file(req):
    """
    Function that takes is a flask request and stores the file in the desired location. 
    If the file is already present, it deleted the existing file and replaces it with a new one

    Args: an object of type flask.request
    Returns: Empty
    """
    if req.method != 'POST': return

    files = req.files.getlist("uploads")

    if req.files:
        for f in req.files: files.append(req.files[f])

    for f in files:

        if not f.filename: continue

        tmp_fname = "temporary"
        while os.path.isfile(tmp_fname):
            tmp_fname = temp_file
        
        TEMP_FPATH = '/Users/libraryuser/Desktop/'
        ACTUAL_FILEPATH = '/Users/libraryuser/Desktop/linear-regression-visualiser'
        f.save(os.path.join(TEMP_FPATH, tmp_fname))

        if os.stat(os.path.join(TEMP_FPATH, tmp_fname)).st_size:
            os.system("mv \"{}\" \"{}\" > /dev/null 2> /dev/null".format(
                os.path.join(TEMP_FPATH, tmp_fname),
                os.path.join(ACTUAL_FILEPATH, f.filename))   
            )
            return f.filename
        else:
            os.remove(os.path.join(TEMP_FPATH, tmp_fname))
            return f.filename
            pass
    


def create_line_plot (dff):
    """
    Function that creates an embedded plotly graph in flask with the feature of hover effect which gives
    the confusion matrix.

    Args: a pandas.DataFrame
    Returns: a JSON String
    """
    data = [
        go.Scatter(
            x = [0,1],
            y = [0,1],
            mode = 'lines',
            line={'dash': 'dash', 'color': '#e8a53a'},
            showlegend = False
        ),
        go.Scatter(
            x = dff['fpr'],
            y = dff['tpr'],
            mode = 'markers+lines',
            line={'dash': 'solid', 'color': 'green'},
            marker = {'size': dff['threshold'],'color':'#ceefe4'},
            text = dff['confusionMTR'],
            hovertemplate = "<br>"+
                "True Positve : %{text[0][0]:,}<br>" +
                "False Positive : %{text[0][1]:,}<br>" +
                "False Negative: %{text[1][0]:,}<br>" +
                "True Negative: %{text[1][1]:,} <br>" +
                "Threshold: %{marker.size:,} <br>" +
                "<extra></extra>",
            showlegend = False,
            textposition = "top left",
            opacity = 0.75
        )
    ]
    layout = go.Layout(
        template = pio.templates['plotly'],
        title=go.layout.Title(
        text='ROC-AUC Curve',
        font=dict(
                family='Courier New, monospace',
                size=40,
                color='#000000'),
        xref='paper',
        x=0),

        xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text='True Positive Rate',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            ))
        ),

        yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text='False Positive Rate',
            font=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )

    )

    fig = go.Figure(data=data,layout=layout)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')
    
@app.route("/result", methods=['POST'])
def result():
    f_name = upload_file(request)
    to_predict_list = request.form.to_dict()
    val_lst = list(to_predict_list.values())
    datafrm = mdl.main_imp(f_name,val_lst[-1])
    temp_json = create_line_plot(datafrm)
    return render_template('index.html', v = temp_json)

if __name__ == '__main__':
    app.run(debug = True)     