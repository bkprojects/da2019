from collections import defaultdict


def createDictionary():
    # Auslesen des gtp Dokuments
    global gtp_document_content
    gtp_filename = '2700270.gtp'
    gtp_document = open(gtp_filename, "r")
    if gtp_document.mode == 'r':
        gtp_document_content = gtp_document.read()

    # Aufsplitten der Wörter und Teilen von den Koordinaten
    gtp_words_arr = []
    split_gtp_doc = gtp_document_content.splitlines()
    split_gtp_doc = [a.split(' ') for a in split_gtp_doc]
    for i in range(len(split_gtp_doc)):
        gtp_words_arr.append(split_gtp_doc[i][4])
    distinct_gtp_words = list(set(gtp_words_arr))

    # Erstellen des dictionaries
    gtp_dictionary = defaultdict(list)
    for i in range(len(gtp_words_arr)):
        gtp_dictionary[gtp_words_arr[i]].append((int(split_gtp_doc[i][0]),
                                                 int(split_gtp_doc[i][1]),
                                                 int(split_gtp_doc[i][2]),
                                                 int(split_gtp_doc[i][3])))
    return gtp_dictionary


def getSelectedWordCoords(dict, word, index=None):
    dict_index = 0
    if not index is None:
        dict_index = index
    (x1, y1, x2, y2) = dict[word][dict_index]
    return x1, y1, x2, y2