import json
import smtplib
import uuid
import os
import glob
import email
from os.path import basename
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate

username = "raspisurveilance2019@gmail.com"
password = "raspberrypi"

class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		# construct the file path
		self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
			rand=str(uuid.uuid4()), ext=ext)

	def cleanup(self):
		# remove the file
		os.remove(self.path)


def send_email():
	fromaddr = username
	# for email_address in conf['email_address']:
	toaddrs  = "sabar.noida@gmail.com"
	# print("[INFO] Emailing to {}".format(email_address))
	text = 'Hey Someone in Your House!!!!'
	subject = 'Security Alert!!'
	message = 'Subject: {}\n\n{}'.format(subject, text)

	msg = MIMEMultipart()
	msg['From'] = fromaddr
	msg['To'] = toaddrs
	msg['Date'] = formatdate(localtime=True)
	msg['Subject'] = subject
	msg.attach(MIMEText(text))



	# set attachments
	files = glob.glob("video*")
	print("[INFO] Number of images attached to email: {}".format(len(files)))
	for f in files:
	    with open(f, "rb") as fil:
	        part = MIMEApplication(
	            fil.read(),
	            Name=basename(f)
	        )
	        part['Content-Disposition'] = 'attachment; filename="%s"' % basename(f)
	        msg.attach(part)

	# Credentials (if needed) : EDIT THIS

	# The actual mail send
	server = smtplib.SMTP('smtp.gmail.com',587)
	server.starttls()
	server.login(username,password)
	server.sendmail(fromaddr, toaddrs, msg.as_string())
	server.quit()



def send_mail(conf, files=None, ):
    assert isinstance(send_to, list)

    msg = MIMEMultipart()
    msg['From'] = "raspisurveilance2019@gmail.com"
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(text))



    smtp = smtplib.SMTP(server)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.close()

# send_email()
