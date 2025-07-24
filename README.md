# GPT2_Reproduce

Before training:
'''
> Hello, I'm a language model, Definitive lined 214 1924 Techniques Jenkins Presbytershadowribe constituent moratorium examine bucket Jacenkowikipedia oakón goalt crucial \"lé
> Hello, I'm a language model,Importproblem'tKnow balcon doomed NB Runesidablelegate dogma Post uncsterdam Lone Indonesian applicant speakovo martialdeep barred
> Hello, I'm a language model, symbolicCommunexistence Adventuresgres Speed message Martialasure Roosevelt bring LG mapped grittyire Constantiggleaddin Matter most Bollclud
> Hello, I'm a language model, detectordisc Wiz separates Franz Leone vicious desertSETasis lur Dres sensitivity Pants CAN757 identifiable Shapiro spontEnglish overwhelminglyConn
> Hello, I'm a language model,book 978Ins challenging Cemetery Qinants adoptedometimes aquatic Discipline unbeaten flakes wasting015 br HobbitMaterials outer rive Garrett venues
'''

After training:
'''
Sample 0 : Hello, I'm a language model, that has been shown in most cases on the web, the language will be the target language of general education. But,
Sample 1 : Hello, I'm a language model, what is a picture on a computer is an embedded system, is that a computer's input is what's been asked to
Sample 2 : Hello, I'm a language model, and using that model to model complex situations using the same model in three dimensions:
One of the most common examples is
Sample 3 : Hello, I'm a language model, but you can find the default settings for an interface. In a previous tutorial on desktop computer, this is a default-
'''

## improvements
for faster training:
1. Add compiler
2. Include Flash Attention
3. add norm
4. Add dynamic learning rate (decay)
5. Add decay to >= 2 dimension parameters for AdamW optimizer

* add feature of expanding small batch size to equivalent of ~0.5M batch size from GPT2 paper while using small core GPU

* Add DDP (Data distributed parallel) feature to enable parallel training on multiple GPU

## dataset
https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
* other datasetes for LLM training: FineWeb_Edu

https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu