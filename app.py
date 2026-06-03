import os

import torch
from flask import request
from flask import session
import uuid


from utils.models import VGG_Encoder,Decoder
from flask import Flask,render_template,redirect,url_for,send_from_directory
from flask_wtf import FlaskForm
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
from utils.utils import adaptive_instance_normalization
from wtforms import FileField,FloatField,HiddenField,SubmitField
from wtforms.validators import InputRequired

from PIL import Image
import io
from torchvision import transforms
app=Flask(__name__)
app.config["SECRET_KEY"]="secret"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app.config["UPLOAD_FOLDER"] = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)
app.config["ALLOWED_EXTENSIONS"]={"jpg","png","jpeg"}
Bootstrap(app)
os.makedirs(app.config["UPLOAD_FOLDER"],exist_ok=True)
class uploadform(FlaskForm):
    content = FileField("content image")
    style = FileField("style image")
    content_path=HiddenField()
    style_path=HiddenField()
    alpha=FloatField("alpha",default=1.0)
    submit=SubmitField("Style Transfer")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder=VGG_Encoder("utils/vgg_normalised.pth").to(device)
decoder=Decoder().to(device)
state_dict = torch.load(
    "decoder_final.pth",
    map_location=torch.device('cpu')
)

new_state_dict = {}

for key, value in state_dict.items():
    new_key = key.replace("net", "decoder")
    new_state_dict[new_key] = value

decoder.load_state_dict(new_state_dict)
encoder.eval()
decoder.eval()
def Allow(file_name):
    return (
        '.' in file_name and
        file_name.rsplit('.',1)[1].lower()
        in app.config["ALLOWED_EXTENSIONS"]
    )
def style_transfer(content_image,style_image,encoder,decoder,alpha,device):
    content_transfrom=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transfrom=transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    t_content_image=content_transfrom(content_image).unsqueeze(0).to(device)
    t_style_image=style_transfrom(style_image).unsqueeze(0).to(device)

    with torch.no_grad():
        style_features=encoder(t_style_image,test=True)
        content_features=encoder(t_content_image,test=True)
        stylized_features = adaptive_instance_normalization(
            content_features,
            style_features
        )
        stylized_features=alpha*stylized_features+(1-alpha)*content_features
        generated_image=decoder(stylized_features)
    return generated_image
def save_image(image,path):
    image=image.cpu().clone()
    image=image.squeeze(0)
    image=image.clamp(0,1)
    image=transforms.ToPILImage()(image)
    image.save(path)

@app.route('/',methods=["GET","POST"])
def index():


    form=uploadform()
    result_image=None
    content_filename=None
    style_filename=None
    error=None
    if form.validate_on_submit():
     if form.content.data and form.content.data.filename:
         if Allow(form.content.data.filename):
            content_filename=secure_filename(form.content.data.filename)
            form.content.data.save(os.path.join(app.config["UPLOAD_FOLDER"],content_filename))
            form.content_path.data=os.path.join(app.config["UPLOAD_FOLDER"],content_filename)
            session["content"] = content_filename
           
         else:
           content_filename = session.get("content")
            
     if form.style.data and form.style.data.filename:
      if Allow(form.style.data.filename):
         style_filename=secure_filename(form.style.data.filename)
         form.style.data.save(os.path.join(app.config["UPLOAD_FOLDER"],style_filename))
         form.style_path.data=os.path.join(app.config["UPLOAD_FOLDER"],style_filename)
         session["style"] = style_filename
      else:
           style_filename = session.get("style")
     if content_filename and style_filename:
         content_path=os.path.join(app.config["UPLOAD_FOLDER"],content_filename)
         style_path=os.path.join(app.config["UPLOAD_FOLDER"],style_filename)
         try:
             content_image=Image.open(content_path).convert("RGB")
             style_image=Image.open(style_path).convert("RGB")
             alpha=float(form.alpha.data)
             styled_image=style_transfer(content_image,style_image,encoder,decoder,alpha,device)
             generated_image_filename = (
                    f"{uuid.uuid4().hex}_stylized.jpg"
            )

             generated_image_filepath=os.path.join(app.config["UPLOAD_FOLDER"],generated_image_filename)
             save_image(styled_image, generated_image_filepath)
             result_image=generated_image_filename
         except Exception as e:
             error=str(e)
             
             
             
   
     

    if request.method == "POST" and not form.validate():
        error = "Something is missing"
    
    return render_template("index.html",form=form,result_image=result_image,content_image=content_filename,style_image=style_filename,error=error)
@app.route("/uploads/<filename>")
def send_image(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],filename)
@app.route("/examples/<path:filename>")
def send_example(filename):
    return send_from_directory("examples",filename)
if __name__=='__main__':
    from werkzeug.serving import    run_simple
    run_simple("localhost",5000,app,use_reloader=True,use_debugger=True)

