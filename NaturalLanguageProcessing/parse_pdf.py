"""
@file parse_pdf.py
@brief parse pdf, usually a technical pdf

@author Ernest Yeung
@email ernestyalumni dot gmail dot com

Remember to import these dependencies:
import pdfminer

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

"""  
import pdfminer

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

from cStringIO import StringIO

# Breakdown of the method to call in order to convert from pdf to txt

def pdf_to_txt(filename):
	rsrcmgr = PDFResourceManager()
	retstr = StringIO()
	codec = 'utf-8'
	laparams = LAParams()
	device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
	interpreter = PDFPageInterpreter(rsrcmgr, device)
	
	password =""
	maxpages =0 
	caching = True
	pagenos=set()
	
	fp = file(filename,'rb')

	pages_grabbed = PDFPage.get_pages(fp,pagenos,maxpages=maxpages,password=password,caching=caching)  # check_extractable=True
	
	for page in pages_grabbed:  
		interpreter.process_page(page)
	
	fp.close()
	device.close()
	ret = retstr.getvalue()	
	ret_decoded = ret.decode('utf-8')	
	return ret_decoded
	


	
	
