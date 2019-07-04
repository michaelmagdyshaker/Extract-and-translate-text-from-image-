from googletrans import Translator
import pyarabic.araby as araby
from spellchecker import SpellChecker

def file_read(fname):
        content_array = []
        with open(fname) as f:
                #Content_list is the list that contains the read lines.     
                for line in f:
                        content_array.append(line)
                print(content_array)
        return content_array
def trans(str):
    translator = Translator()
    translator = Translator(service_urls=[
          'translate.google.com',
          'translate.google.co.kr'])


    content_array=file_read('test.txt')
    spell = SpellChecker()
    str=spell.correction(str)
    print(str)
    f = open("testar.txt",'a',encoding='utf-8')
                   
    translations = translator.translate(str, dest='ar')
    #for translation in translations:
    print(translations.origin )
    print((translations.text).encode('utf8'))
    f.write((translations.text))
    f.write(' \n')
    f.close()

    return translations.text


#spell = SpellChecker()

## find those words that may be misspelled
#misspelled = spell.unknown(['something', 'is', 'hapenning', 'here'])

#for word in misspelled:
#    # Get the one most likely answer
#    print(spell.correction(word))

#    # Get a list of likely options
#    print(spell.candidates(word))