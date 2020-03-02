import os
from flask import Flask, request, render_template, url_for, redirect

app = Flask(__name__)

@app.route("/")
def fileFrontPage():
    return render_template('fileform.html')

@app.route("/handleUpload", methods=['POST'])
def handleFileUpload():
    # create a part of the form which takes in the user-name 
    if 'photo' in request.files:
        photo = request.files['photo']
        photo.save(photo.filename)
        if photo.filename != '':            
            photo.save(os.path.join('/Users/libraryuser/Desktop/linear-regression-visualiser', photo.filename))
    # return redirect(url_for('fileFrontPage'))
    return "File uploaded succesfully"

if __name__ == '__main__':
    app.run(debug = True)     