from flask import Flask,render_template,request,Response
from tools import *
from werkzeug.utils import secure_filename
import json
import conf
import os
app = Flask(__name__)

user = conf.user
path = conf.path

@app.route('/')
def use():
    return render_template('choix_user.html')

@app.route('/choix', methods=['POST'])
def choix():
        #try:
        choix = request.form.get('choix').strip()
        if choix == "pred":
            l=[]
            for fichier in os.listdir(conf.path_model):
                if "clf" in fichier:
                    l.append(fichier)
            return render_template('prediction.html',models=l)
        if choix == "train":
            return render_template('entrain.html')
        #except:
        return render_template('choix_user.html', mess="Reformuler votre choix")




@app.route('/prediction_inter', methods=['POST'])
def prediction_inter():
    text = request.form.get('mbappe').strip()
    model = request.form.get('model').strip()
    print(choix)
    d={"0": "Texte à connotation négative", "1":"Texte à connotation positive"}
    pred=d[str(int(prediction_user(text,model)[0]))]

    return render_template('resultat.html', prediction=pred, model=choix)


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
       user = request.form.get('user').strip()
       f = request.files['file']
       f.save("data/"+user+'_'+secure_filename(f.filename))
       path ="data/"+secure_filename(f.filename)
       df = lecture(path)
       df = macronet(df)
       df = equilibre(df)
       df['content_nett'] = df['content'].apply(nett)
       x = feature(df,user)
       y = df['score']
       met = train(x, y,user)

       return render_template('train.html',comments='Fichier chargé et entrainement réalisé. Les metrics sont {}'.format(met))


if __name__ == "__main__":
    app.run(port="5000",debug=True)
    
    

