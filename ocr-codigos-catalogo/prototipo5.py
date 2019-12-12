import pytesseract
import pdf2image
from PyPDF2 import PdfFileReader
import re
from pathlib import Path
import itertools


def pdf2text(file_path, page_range=None, tessconf=None, text_layer=False):
    """Convert pdf to text.
    
    Parameters
    ----------
    file_path : str
        Path to pdf file.
        
    page_range : list of int, default None
        List with the page numbers to process.
        
    tessconf : str, default None
        Configurations used to invoke Tesseract OCR.
        
    text_layer : bool, default False
        Whether to assume text layer in pdf or not.
        If set to true it extracts the text layer and omits OCR completely,
        improving performance.
        
    Returns
    -------
    texts : dict of str
        Dictionary with each page of text.
    """
    file_path = Path(file_path)
    pdf_file = PdfFileReader(file_path.as_posix())
    n_pages = pdf_file.getNumPages()

    if page_range is None:
        page_range = range(1, n_pages + 1)

    if text_layer:
        texts = {
            k: pdf_file.getPage(k - 1).extractText()
            for k in page_range}
    else:
        texts = dict()
        for k in page_range:
            print("Processing page {}".format(k))
            image = pdf2image.convert_from_path(
                file_path, dpi=300, grayscale=False,
                first_page=k, last_page=k)[0]
            text = pytesseract.image_to_string(
                image, config=tessconf)
            texts[k] = text

    return texts


def load_codes(valid_codes_path):
    """Load valid codes.
    
    Parameters
    ----------
    valid_codes_path : str
        Path to text file containing valid product codes.
    
    Returns
    -------
    valid_codes : set of str
        Set with the valid product codes.
    
    """
    valid_codes_file = open(valid_codes_path)
    valid_codes_str = valid_codes_file.read()
    valid_codes = set(filter(None, valid_codes_str.split("\n")))
    valid_codes_file.close()
    return valid_codes


def list_substitutions(st, substitutions):
    """List of variations obtained from string by substitution of chars.

    Parameters
    ----------
    st : str

    substitutions : dict of str
        Dict with substitutions corresponding to each key.

    Returns
    -------
    sub_list : list of str

    """
    sub_space = [substitutions[c] if c in substitutions else c for c in st]
    sub_list = ["".join(sub) for sub in itertools.product(*sub_space)]
    return sub_list


def expand_with_substitutions(codes, substitutions):
    """Expand codes with substitutions.

    Parameters
    ----------
    codes : set of str

    substitutions : dict of str
        Dict with substitutions corresponding to each key.

    Returns
    -------
    extended_codes : list of str

    """
    extended_codes = []
    for code in codes:
        extended_codes.extend(list_substitutions(code, substitutions))
    return extended_codes


def search_codes(texts, valid_codes, code_pattern=r'\w*', code_header="",
                 show_tentative=False, substitutions=None):
    """Search for possible product codes in text.
    
    Parameters
    ----------
    texts : dict of str
        Dict with the texts for each page.
    
    valid_codes : set of str
        Set with the valid product codes.
        
    code_pattern : str, optional
        Regex with the code pattern to search in the text. Defaults to r'\w*'.

    code_header : str, optional
        Regex with the code header pattern to consider when searching. Defaults to "".

    show_tentative : bool, optional
        If true, return a dict with the tentative codes found before validation. Defaults to False.

    substitutions : dict of str, optional
        Dictionary with the mapping to use when expanding the codes with substitutions.
        The keys contain characters that might be confused by OCR, and the values contain
        the characters that might actually correspond to the key.
        For example, if mapping = {"O":"0O", "l":"1I"}, that means that a "O" found in a code
        might actually be a "0" or an "O", while a "l" might actually be a "1" or an "I".
        Note that in the first case, we are implying that a "O" might not be an error,
        while in the second case, we are implying that "l" is always an error. Defaults to None.
        
    Returns
    -------
    codes_found : dict of sets
        Dict with the set of product codes found, for each page.

    tentative_codes : dict of sets, optional
        Dict with the set of tentative codes.
    """
    tentative_codes = dict()
    codes_found = dict()
    search_pattern = code_header + "(" + code_pattern + ")"

    for page in texts.keys():
        matches = re.finditer(search_pattern, texts[page])
        tentative_codes[page] = {match.group(1) for match in matches}
        if substitutions is not None:
            tentative_codes[page] = set(expand_with_substitutions(tentative_codes[page], substitutions))
        codes_found[page] = tentative_codes[page].intersection(valid_codes)

    if show_tentative:
        return codes_found, tentative_codes

    return codes_found


def detect_codes(
        file_path, valid_codes_path, code_pattern=r'\w*', code_header="",
        page_range=None, text_layer=False, show_tentative=False, substitutions=None,
        lang='eng', oem=3, psm=3, extra_config=""):
    """Detect valid product codes in pdf file.
    
    Parameters
    ----------
    file_path : str
        Path to pdf file.
        
    valid_codes_path: str
        Path to valid codes file.
    
    code_pattern : str, optional
        Regex with the code pattern to search in the text. Defaults to r'\w*'.

    code_header : str, optional
        Regex with the code header pattern to consider when searching. Defaults to "".

    page_range : list of int, optional
        List with the page numbers to process. If no range is given then process all the pages.
        Defaults to None.
        
    text_layer : bool, optional
        Whether to assume text layer in pdf or not.
        If set to true it extracts the text layer and omits OCR completely,
        improving performance. Defaults to False.

    show_tentative : str, optional
        If true, return a dict with the tentative codes found before validation. Defaults to False.

    substitutions : dict of str, optional
        Dictionary with the mapping to use when expanding the codes with substitutions.
        The keys contain characters that might be confused by OCR, and the values contain
        the characters that might actually correspond to the key.
        For example, if mapping = {"O":"0O", "l":"1I"}, that means that a "O" found in a code
        might actually be a "0" or an "O", while a "l" might actually be a "1" or an "I".
        Note that in the first case, we are implying that a "O" might not be an error,
        while in the second case, we are implying that "l" is always an error.
        
    lang : str, optional
        Language to use in Tesseract OCR. Defaults to 'eng'.
        
    oem : int, optional
        Specify OCR Engine Mode. The options are:
            0 = Original Tesseract only.
            1 = Neural nets LSTM only.
            2 = Tesseract + LSTM.
            3 = Default, based on what is available.
        
    psm : int, optional
        0 = Orientation and script detection (OSD) only.
        1 = Automatic page segmentation with OSD.
        2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
        3 = Fully automatic page segmentation, but no OSD. (Default)
        4 = Assume a single column of text of variable sizes.
        5 = Assume a single uniform block of vertically aligned text.
        6 = Assume a single uniform block of text.
        7 = Treat the image as a single text line.
        8 = Treat the image as a single word.
        9 = Treat the image as a single word in a circle.
        10 = Treat the image as a single character.
        11 = Sparse text. Find as much text as possible in no particular order.
        12 = Sparse text with OSD.
        13 = Raw line. Treat the image as a single text line,
             bypassing hacks that are Tesseract-specific.

    extra_config: str
        Controls extra configuration parameters for Tesseract. String must be in the format "-c CONFIGVAR=VALUE".
        Multiple -c arguments are allowed. For a list of parameters supported by tesseract see docs/params.txt.
        Note that some parameters are supported only in legacy mode (oem 0), for example, "tessedit_char_whitelist".
             
    Returns
    -------
    codes_found : dict of sets
        Dict with the set of product codes found, for each page.

    tentative_codes : dict of sets, optional
        Dict with the set of tentative codes.
    """
    valid_codes_path = Path(valid_codes_path)
    valid_codes = load_codes(valid_codes_path)

    tessconf = " --tessdata-dir ./tessdata {} --user-words {} -l {} --oem {} --psm {}".format(
        extra_config, valid_codes_path.absolute().as_posix(), lang, oem, psm, )

    texts = pdf2text(file_path, page_range=page_range, text_layer=text_layer, tessconf=tessconf)

    codes_found = search_codes(texts, valid_codes, code_pattern, code_header=code_header,
                               show_tentative=show_tentative, substitutions=substitutions)
    return codes_found


def main(test):
    args = load_test(test)
    codes_found, tentative_codes = detect_codes(*args)

    print("Tentative codes found")
    for k, tentative in tentative_codes.items():
        print(k, tentative)
    print("Product codes found:")
    for k, codes in codes_found.items():
        print(k, codes)

    n_codes_found = len(list(itertools.chain(*codes_found.values())))
    print("Found {} codes".format(n_codes_found))


def load_test(test):
    if test == 1:
        file_path = './data/carasaga/Carasaga.pdf'
        codes_path = './data/carasaga/valid-codes.txt'
        code_header = 'R.f:\s*'
        code_pattern = r'\d{5}'
        page_range = None
        text_layer = False
        oem = 3
        psm = 3
        lang = 'eng'
        extra_config = ""
        show_tentative = True
        substitutions = None
    elif test == 2:
        file_path = './data/carasaga/Carasaga.pdf'
        codes_path = './data/carasaga/valid-codes.txt'
        code_header = 'R.f:\s*'
        code_pattern = r'\d{5}'
        page_range = None
        text_layer = True
        oem = 3
        psm = 3
        lang = 'eng'
        extra_config = ""
        show_tentative = True
        substitutions = None
    elif test == 3:
        file_path = './data/sdm-noel/SdM-Noel2018.pdf'
        codes_path = './data/sdm-noel/valid-codes.txt'
        code_header = 'REF.\s'
        code_pattern = r'[A-Z]{2,3}\d{2}[A-Z]?'
        page_range = None
        text_layer = True
        oem = 3
        psm = 3
        lang = 'eng'
        extra_config = ""
        show_tentative = True
        substitutions = None
    else:
        file_path = './data/sdm2018/SDM2018.pdf'
        codes_path = './data/sdm2018/valid-codes.txt'
        code_header = 'r.f.\s*'
        code_pattern = r'[A-Z0-9]{4,5}'
        #page_range = range(9, 30)
        page_range = None
        text_layer = False
        lang = 'eng'
        oem = 3
        psm = 3
        extra_config = ""
        show_tentative = True
        substitutions = {"O": "O0", "0": "O0", "l": "1I", "1": "1I", "I": "1I", "T": "T1"}

    return (file_path, codes_path, code_pattern, code_header, page_range,
            text_layer, show_tentative, substitutions,
            lang, oem, psm, extra_config)


if __name__ == "__main__":
    main(0)
