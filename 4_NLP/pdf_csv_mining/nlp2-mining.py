import re, PyPDF2

def extract_abstract(pdf_fname = 'sample.pdf'):  # sample.pdf 파일을 준비한다. 이 경우는 논문의 ABSTRACT 부분을 추출하는 것으로 코딩되었다.
    pdf_file = open(pdf_fname, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)

    first_page = pdf_reader.pages[0]
    text = first_page.extract_text()

    abstract_start = text.find('Abstract')  # abstract keyword 검색
    if abstract_start == -1:
        abstract_start = text.find('ABSTRACT')

    abstract = ''
    if abstract_start == -1:
        return abstract

    abstract_start += len('Abstract')
    introduction_start = text.find('Introduction')  # introduction keyword 검색
    if introduction_start == -1:
        introduction_start = text.find('Background')
    if introduction_start == -1:
        return abstract
    introduction_start -= 4

    abstract_end = introduction_start if introduction_start != -1 else len(text)
    abstract = text[abstract_start:abstract_end]

    # remove new line characters
    abstract = abstract.replace('\n', ' ')

    # remove digits and special characters
    # abstract = re.sub(r'[^a-zA-Z0-9().,% ]', '', abstract)
    print(abstract)

extract_abstract()