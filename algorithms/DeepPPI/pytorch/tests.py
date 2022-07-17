from deep import *
from data import *


def test_tensorize():
    '''Test creation of tensors '''
    fichier = "data0118.txt"
    donnees = get_lines_from_file(fichier)
    all_letters_to_numbers(donnees)
    print(tensorize(donnees[2][2], donnees[2][3]))

    
def test_tensorize_all():
    fichier = "data0118.txt"
    donnees = get_lines_from_file(fichier)
    all_letters_to_numbers(donnees)
    print(tensorize_all(donnees)) 

    
def test_all_letters_to_numbers():
    '''Test for the all_letters_to_numbers function'''
    fichier = "data0118.txt"
    donnees = get_lines_from_file(fichier)
    all_letters_to_numbers(donnees)
    print(donnees[2][2])
    print(donnees[2][3])
    
    
def test_max_size_sequence():
    '''Test finding max size'''
    fichier = "data0118.txt"
    donnees = get_lines_from_file(fichier)
    all_letters_to_numbers(donnees)
    print(max_size_sequence(donnees))

def test_one_hotting():
    string = 'MAARVAAVRAAAWLLLGAATGLTRGPAAAFTAARSDAGIRAMCSEIILRQEVLKDGFHRDLLIKVKFGESIEDLHTCRLLIKQDIPAGLYVDPYELASLRERNITEAVMVSENFDIEAPNYLSKESEVLIYARRDSQCIDCFQAFLPVHCRYHRPHSEDGEASIVVNNPDLLMFCDQEFPILKCWAHSEVAAPCALENEDICQWNKMKYKSVYKNVILQVPVGLTVHTSLVCSVTLLITILCSTLILVAVFKYGHFSL'
    for i in range(len(string)):
        print(one_hotting(string[i]))
        
test_one_hotting()
