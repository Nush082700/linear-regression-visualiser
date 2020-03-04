import os
from flask import Flask, request, render_template, url_for, redirect
import pickle
import model as mdl
import model_viz as viz
import plotly
import plotly.graph_objs as go

import pandas as pd
import numpy as np
import json


app = Flask(__name__)

#Helper Functions begin here
def upload_file(req):
    if req.method != 'POST': return

    # file with 'multiple' attribute is 'uploads'
    files = req.files.getlist("uploads")

    # get other single file attachments
    if req.files:
        for f in req.files: files.append(req.files[f])

    for f in files:

        if not f.filename: continue

        tmp_fname = "temporary"
        while os.path.isfile(tmp_fname):
            # you can never rule out the possibility of similar random 
            # names
            tmp_fname = temp_file
        
        TEMP_FPATH = '/Users/libraryuser/Desktop/'
        ACTUAL_FILEPATH = '/Users/libraryuser/Desktop/linear-regression-visualiser'
        # TEMP_FPATH is he root temporary directory
        f.save(os.path.join(TEMP_FPATH, tmp_fname))

        # and here comes the trick
        if os.stat(os.path.join(TEMP_FPATH, tmp_fname)).st_size:
            os.system("mv \"{}\" \"{}\" > /dev/null 2> /dev/null".format(
                os.path.join(TEMP_FPATH, tmp_fname),
                os.path.join(ACTUAL_FILEPATH, f.filename))   
            )
            return f.filename
        else:
            # cleanup
            os.remove(os.path.join(TEMP_FPATH, tmp_fname))
            return f.filename
            pass
    

# def create_plot(fpr,tpr):
#     # N = 40
#     # x = np.linspace(0, 1, N)
#     # y = np.random.randn(N)
#     # df = pd.DataFrame({'fpr': fpr, 'tpr': tpr}) # creating a sample dataframe

#     # print("the data frame is")
#     # print(df)

#    fig = go.Figure(
#     data=[go.Bar(y=[2, 1, 3])],
#     layout_title_text="A Figure Displayed with the 'svg' Renderer"
#     )
#     fig.show(renderer="svg")

#     # graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
#     # print("the type of return is")
#     # print(type(graphJSON))
#     # return graphJSON
# # Helper functions end here


@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')
    
@app.route("/result", methods=['POST'])
def result():
    f_name = upload_file(request)
    print(f_name)
    # create a part of the form which takes in the user-name 
    # if 'photo' in request.files:
    #     photo = request.files['photo']
    #     photo.save(photo.filename)
    #     if photo.filename != '':            
    #         photo.save(os.path.join('/Users/libraryuser/Desktop/linear-regression-visualiser', photo.filename))
    
    to_predict_list = request.form.to_dict()
    val_lst = list(to_predict_list.values())
    # print(val_lst)
    # model = pickle.load(open("model.pkl","rb"))
    fpr,tpr = mdl.main_imp(f_name,val_lst[-1])
    # score = mdl.main_imp(f_name,val_lst[-1])
    # prediction = "Yay! It worked"
    # scp,div1 = viz.make_plot()
    # return render_template("graph.html",script=scp, div=div1)
    # return create_plot(fpr,tpr)
    fig = go.Figure(
    data=[go.Bar(y=[2, 1, 3])],
    layout_title_text="A Figure Displayed with the 'svg' Renderer"
    )
    fig.show(renderer="chrome")
    # return render_template('index.html', plot = line) #this has changed
    # return render_template("result.html",prediction=score)

    # return redirect(url_for('fileFrontPage'))

if __name__ == '__main__':
    app.run(debug = True)     