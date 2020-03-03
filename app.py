import os
from flask import Flask, request, render_template, url_for, redirect
import pickle
import model as mdl

app = Flask(__name__)

@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')


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
    print(val_lst)
    # model = pickle.load(open("model.pkl","rb"))
    score = mdl.main_imp(f_name,val_lst[-1])
    # prediction = "Yay! It worked"
    return render_template("result.html",prediction=score)

    # return redirect(url_for('fileFrontPage'))

if __name__ == '__main__':
    app.run(debug = True)     