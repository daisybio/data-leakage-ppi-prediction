#ifndef GLOBAL_PARAMETERS_H_
#define GLOBAL_PARAMETERS_H_
#include <string.h>
#include <stdint.h>
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <set>
#include <algorithm>
#ifdef PARAL
#include <omp.h>
#endif
#include <time.h>
#include <sstream>
using namespace std;
int PROTEOME = 0;
string PROTEIN_FN;
string HSP_FN;
string TRAIN_FN;
string TEST_FN_POS;
string TEST_FN_NEG;
string OUTPUT_FN;
string OUTPUT_pos_FN;
string OUTPUT_neg_FN;
int T_hsp_max = 40; //Tdom
int DEBUG = 0;
int STATS = 1;

const int NPRO = 20117; //Park_Marcotte_human

int num_protein = 0;
int k_size = 20;
map <int, string> p_id_name;
map <string, int> p_name_id;
map <int, string> p_id_seq; //giving a protein id, return the protein sequence of it
//read protein sequences, fill the two maps -- p_id_name and p_name_id
const int BLOSUM80[23][23] =	//it's acutally PAM120, the last row and colum indicate "X"
    {
		{  3, -3, -1,  0, -3, -1,  0,  1, -3, -1, -3, -2, -2, -4,  1,  1,  1, -7, -4,  0,  0, -1, -1 },
		{ -3,  6, -1, -3, -4,  1, -3, -4,  1, -2, -4,  2, -1, -5, -1, -1, -2,  1, -5, -3, -2, -1, -2 },
		{ -1, -1,  4,  2, -5,  0,  1,  0,  2, -2, -4,  1, -3, -4, -2,  1,  0, -4, -2, -3,  3,  0, -1 },
		{  0, -3,  2,  5, -7,  1,  3,  0,  0, -3, -5, -1, -4, -7, -3,  0, -1, -8, -5, -3,  4,  3, -2 },
		{ -3, -4, -5, -7,  9, -7, -7, -4, -4, -3, -7, -7, -6, -6, -4,  0, -3, -8, -1, -3, -6, -7, -4 },
		{ -1,  1,  0,  1, -7,  6,  2, -3,  3, -3, -2,  0, -1, -6,  0, -2, -2, -6, -5, -3,  0,  4, -1 },
		{  0, -3,  1,  3, -7,  2,  5, -1, -1, -3, -4, -1, -3, -7, -2, -1, -2, -8, -5, -3,  3,  4, -1 },
		{  1, -4,  0,  0, -4, -3, -1,  5, -4, -4, -5, -3, -4, -5, -2,  1, -1, -8, -6, -2,  0, -2, -2 },
		{ -3,  1,  2,  0, -4,  3, -1, -4,  7, -4, -3, -2, -4, -3, -1, -2, -3, -3, -1, -3,  1,  1, -2 },
		{ -1, -2, -2, -3, -3, -3, -3, -4, -4,  6,  1, -3,  1,  0, -3, -2,  0, -6, -2,  3, -3, -3, -1 },
		{ -3, -4, -4, -5, -7, -2, -4, -5, -3,  1,  5, -4,  3,  0, -3, -4, -3, -3, -2,  1, -4, -3, -2 },
		{ -2,  2,  1, -1, -7,  0, -1, -3, -2, -3, -4,  5,  0, -7, -2, -1, -1, -5, -5, -4,  0, -1, -2 },
		{ -2, -1, -3, -4, -6, -1, -3, -4, -4,  1,  3,  0,  8, -1, -3, -2, -1, -6, -4,  1, -4, -2, -2 },
		{ -4, -5, -4, -7, -6, -6, -7, -5, -3,  0,  0, -7, -1,  8, -5, -3, -4, -1,  4, -3, -5, -6, -3 },
		{  1, -1, -2, -3, -4,  0, -2, -2, -1, -3, -3, -2, -3, -5,  6,  1, -1, -7, -6, -2, -2, -1, -2 },
		{  1, -1,  1,  0,  0, -2, -1,  1, -2, -2, -4, -1, -2, -3,  1,  3,  2, -2, -3, -2,  0, -1, -1 },
		{  1, -2,  0, -1, -3, -2, -2, -1, -3,  0, -3, -1, -1, -4, -1,  2,  4, -6, -3,  0,  0, -2, -1 },
		{ -7,  1, -4, -8, -8, -6, -8, -8, -3, -6, -3, -5, -6, -1, -7, -2, -6, 12, -2, -8, -6, -7, -5 },
		{ -4, -5, -2, -5, -1, -5, -5, -6, -1, -2, -2, -5, -4,  4, -6, -3, -3, -2,  8, -3, -3, -5, -3 },
		{  0, -3, -3, -3, -3, -3, -3, -2, -3,  3,  1, -4,  1, -3, -2, -2,  0, -8, -3,  5, -3, -3, -1 },
		{  0, -2,  3,  4, -6,  0,  3,  0,  1, -3, -4,  0, -4, -5, -2,  0,  0, -6, -3, -3,  4,  2, -1 },
		{ -1, -1,  0,  3, -7,  4,  4, -2,  1, -3, -3, -1, -2, -6, -1, -1, -2, -7, -5, -3,  2,  4, -1 },
		{ -1, -2, -1, -2, -4, -1, -1, -2, -2, -1, -2, -2, -2, -3, -2, -1, -1, -5, -3, -1, -1, -1, -2 },
	} ;
char num_to_letter[25] = {'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',  'B', 'Z', 'X', 'U', 'O'};

void load_protein(string protein_fn){
	ifstream fin(protein_fn.c_str());
	if(!fin){
		cout<<"Loading protein file failed"<<protein_fn<<endl;
		exit(3);
	}
	string temp_pName;
	string temp_pSeq;
	int i = 0;
	while(fin>>temp_pName>>temp_pSeq){
		p_id_name[i] = temp_pName.substr(1);
		p_name_id[temp_pName.substr(1)] = i;
		p_id_seq[i] = temp_pSeq;
		i ++;
	}
	fin.close();
	num_protein = i;
}
//from protein id to protein name
string PIDtoName(int id){
	return p_id_name[id];
}

// from protein name to protein id
int PNametoID(string name){
	return p_name_id[name];
}

map <char, int> BLOSUM_convert;
void load_BLOSUM_convert(map <char, int> & BLOSUM_convert){
	BLOSUM_convert['A'] = 0;
	BLOSUM_convert['a'] = 0;
	BLOSUM_convert['r'] = 1;
	BLOSUM_convert['R'] = 1;
	BLOSUM_convert['n'] = 2;
	BLOSUM_convert['N'] = 2;
	BLOSUM_convert['d'] = 3;
	BLOSUM_convert['D'] = 3;
	BLOSUM_convert['c'] = 4;
	BLOSUM_convert['C'] = 4;
	BLOSUM_convert['q'] = 5;
	BLOSUM_convert['Q'] = 5;
	BLOSUM_convert['e'] = 6;
	BLOSUM_convert['E'] = 6;
	BLOSUM_convert['g'] = 7;
	BLOSUM_convert['G'] = 7;
	BLOSUM_convert['h'] = 8;
	BLOSUM_convert['H'] = 8;
	BLOSUM_convert['i'] = 9;
	BLOSUM_convert['I'] = 9;
	BLOSUM_convert['l'] = 10;
	BLOSUM_convert['L'] = 10;
	BLOSUM_convert['k'] = 11;
	BLOSUM_convert['K'] = 11;
	BLOSUM_convert['m'] = 12;
	BLOSUM_convert['M'] = 12;
	BLOSUM_convert['f'] = 13;
	BLOSUM_convert['F'] = 13;
	BLOSUM_convert['p'] = 14;
	BLOSUM_convert['P'] = 14;
	BLOSUM_convert['s'] = 15;
	BLOSUM_convert['S'] = 15;
	BLOSUM_convert['t'] = 16;
	BLOSUM_convert['T'] = 16;
	BLOSUM_convert['w'] = 17;
	BLOSUM_convert['W'] = 17;
	BLOSUM_convert['y'] = 18;
	BLOSUM_convert['Y'] = 18;
	BLOSUM_convert['v'] = 19;
	BLOSUM_convert['V'] = 19;
	BLOSUM_convert['b'] = 20;//all amino acids that are not the 20 standard one are converted into the 21st one
	BLOSUM_convert['B'] = 20;
	BLOSUM_convert['z'] = 20;
	BLOSUM_convert['Z'] = 20;
	BLOSUM_convert['x'] = 20;
	BLOSUM_convert['X'] = 20;
	BLOSUM_convert['u'] = 20;
	BLOSUM_convert['U'] = 20;
	BLOSUM_convert['o'] = 20;
	BLOSUM_convert['O'] = 20;
}
//return the BLOSUM score for a given pair, if they are letters
inline int BLOSUM_score(char x, char y){
	return BLOSUM80[BLOSUM_convert[x]][BLOSUM_convert[y]];
}

//return the BLOSUM score for a given pair, if they are numbers
inline int BLOSUM_score(int x, int y){
	if( (x > 22) || (y > 22) ) {cout<<"parameters out of range for BLOSUM80"<<endl; exit(2);}
	return BLOSUM80[x][y];
}

inline int score_of_kmer(int p1_id, int p2_id, int p1_sta, int p2_sta){
	int score = 0;
	for(int a = 0; a < k_size; a ++){
		score += BLOSUM_score(p_id_seq.at(p1_id).at(a + p1_sta), p_id_seq.at(p2_id).at(a + p2_sta));
	}
	return score;
}

inline int compare_two_strings(string str1, string str2){	//calculate the BLOSUM80 score for two strings
	int score = 0;
	for(uint32_t a = 0; a < str1.length(); a ++){
		score += BLOSUM_score(str1[a], str2[a]);
	}
	return score;
}
vector <vector<int> > hsp_flag;// 2D matrix, each hsp_flag[a][b] indicates in protein a, position b, the number of hsps involves
void init_flag(){
	for(int a = 0; a < num_protein; a ++){
		hsp_flag.push_back(vector<int>());
	}

	for(int a = 0; a < num_protein; a ++){
		for(int b = 0; b < (int)(p_id_seq.at(a).length()); b ++){
			hsp_flag.at(a).push_back(0);
		}
	}
}
void print_flag(){
	string flag_fn = OUTPUT_FN + ".flag";
	ofstream fout(flag_fn.c_str());
	for(int a = 0; a < num_protein; a ++){
		fout<<">"<<p_id_name.at(a)<<"\n";
		fout<<p_id_seq.at(a)<<endl;	
		for(int b = 0; b < (int)(p_id_seq.at(a).length()); b ++){
			fout<<hsp_flag.at(a).at(b)<<" ";
		}
		fout<<endl;
	}
	fout.close();
}

int socre_between_two_hsp(int p1_id, int p2_id, int p1_sta, int p2_sta, int len){
	int total_score = 0;
	for(int a = 0; a < len - k_size + 1; a ++){
		total_score += score_of_kmer(p1_id, p2_id, p1_sta + a, p2_sta + a);
	}
	return total_score;
}
#endif /* GLOBAL_PARAMETERS_H_ */
