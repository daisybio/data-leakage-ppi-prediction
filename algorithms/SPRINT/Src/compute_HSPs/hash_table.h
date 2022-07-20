#ifndef HASH_TABLE_H_
#define HASH_TABLE_H_

#include "global_parameters.h"
struct smer_detail{
	int pro_id;
	int sta_pos;
	//int end_pos;
};
struct hash_entry{
	uint64_t smer;
	vector <smer_detail> smer_occs;
};

class HASH_TABLE{
public:
	string seed;
	uint32_t seed_len;
	uint64_t seed_64;
	uint64_t ht_size;
	hash_entry * hash_table;
	void creat_hash_table(string seed);	//initialize the hash table
	HASH_TABLE();
	void get_hash_size();	// based on the seed length, get the intial size of the hash table
	void convert_seed();
	void encode_pro();	// encode protein, and based on seed, convert it to smers
	void insert_smer_to_ht(uint64_t new_smer, int sta, int pro_id);	//add smer to the hash table
	uint64_t get_ht_index(uint64_t ind, uint64_t smer);
	uint64_t search_in_ht(uint64_t index, uint64_t target_smer);
};
void HASH_TABLE::get_hash_size(){
	seed_len = seed.length();
	ht_size = 0;
	for(boost::unordered_map<int, string> :: iterator it = p_id_seq.begin(); it != p_id_seq.end(); it ++){
		ht_size = (it->second).length() + ht_size - seed_len + 1;
	}
	// select from the large prime integer list
    for(int a = 0; a < 440; a++)
    {
        if(hashTableSizes[a] > (2 * ht_size))
        {
        	ht_size = hashTableSizes[a];
            break;
        }
    }
}
HASH_TABLE :: HASH_TABLE(){

}
void HASH_TABLE :: creat_hash_table(string input_seed){
	#ifdef PARAL
	#pragma omp critical(writeFile)
	#endif
	seed = input_seed;
	this->get_hash_size();
	this->convert_seed();

	hash_table = (hash_entry *) malloc(sizeof(hash_entry) * ht_size);
	for(uint64_t a = 0; a < ht_size; a ++){	//initialization of hash_table, set the smer as 1090715534754863 as default, since no smer would have 25 digit 1
		hash_table[a].smer = hash_table_default;
	}
	this->encode_pro();

	//counting the number of useful entries in the hashtable
	int useful_hash_entry = 0;
	for(uint64_t a = 0; a < ht_size; a ++){	//initialization of hash_table, set the smer as 1090715534754863 as default, since no smer would have 25 digit 1
		if(hash_table[a].smer != hash_table_default){
			useful_hash_entry ++;
		}
	}
}
void HASH_TABLE :: convert_seed(){
	seed_64 = 0;
	for(uint32_t a = 0; a < seed.length(); a ++){
		if(seed[a] == '1'){	//1, add 11111
			seed_64 = (seed_64 << 5);
			seed_64 = (seed_64 | 31);
		}
		else{	// 0, add 00000
			seed_64 = (seed_64 << 5);
		}
	}
}
void HASH_TABLE :: encode_pro(){
	uint64_t temp_smer;		//smer for each seed_length sequence
	uint64_t temp_seq64;	//store the seed_length sequence
	int num_insertion = 0;
	for(boost::unordered_map <int, string> :: iterator it = p_id_seq.begin(); it != p_id_seq.end(); it ++){
		temp_seq64 = 0;
		temp_smer = 0;
		for(uint32_t i = 0; i < (it->second).length(); i ++){
			temp_seq64 = (temp_seq64 << 5);
			switch( (it->second).at(i)){
			case 'a':
			case 'A':
				temp_seq64 = (temp_seq64 | 0);
				break;
			case 'r':
			case 'R':
				temp_seq64 = (temp_seq64 | 1);
				break;
			case 'n':
			case 'N':
				temp_seq64 = (temp_seq64 | 2);
				break;
			case 'd':
			case 'D':
				temp_seq64 = (temp_seq64 | 3);
				break;
			case 'c':
			case 'C':
				temp_seq64 = (temp_seq64 | 4);
				break;
			case 'q':
			case 'Q':
				temp_seq64 = (temp_seq64 | 5);
				break;
			case 'e':
			case 'E':
				temp_seq64 = (temp_seq64 | 6);
				break;
			case 'g':
			case 'G':
				temp_seq64 = (temp_seq64 | 7);
				break;
			case 'h':
			case 'H':
				temp_seq64 = (temp_seq64 | 8);
				break;
			case 'i':
			case 'I':
				temp_seq64 = (temp_seq64 | 9);
				break;
			case 'l':
			case 'L':
				temp_seq64 = (temp_seq64 | 10);
				break;
			case 'k':
			case 'K':
				temp_seq64 = (temp_seq64 | 11);
				break;
			case 'm':
			case 'M':
				temp_seq64 = (temp_seq64 | 12);
				break;
			case 'f':
			case 'F':
				temp_seq64 = (temp_seq64 | 13);
				break;
			case 'p':
			case 'P':
				temp_seq64 = (temp_seq64 | 14);
				break;
			case 's':
			case 'S':
				temp_seq64 = (temp_seq64 | 15);
				break;
			case 't':
			case 'T':
				temp_seq64 = (temp_seq64 | 16);
				break;
			case 'w':
			case 'W':
				temp_seq64 = (temp_seq64 | 17);
				break;
			case 'y':
			case 'Y':
				temp_seq64 = (temp_seq64 | 18);
				break;
			case 'v':
			case 'V':
				temp_seq64 = (temp_seq64 | 19);
				break;
			case 'b':
			case 'B':
			case 'z':
			case 'Z':
			case 'x':
			case 'X':
			case 'u':
			case 'U':
			case 'o':
			case 'O':
				temp_seq64 = (temp_seq64 | 20);
				break;								
			default:
				temp_seq64 = (temp_seq64 | 31);	//letters which other than 25 amino acids, will be encoded by a largest 5-digit number
				cout<<"protein seqeunces contains letter(s) other than the 25 Amino Acids(including U and O) as shown here: http://blast.advbiocomp.com/blast-1.3/blast/PAM120"<<endl;
				cout<<"P_id: "<<it->first<<endl;
				cout<<"position: "<<i<<endl;
				exit(14);
				break;
			}
			temp_smer = (temp_seq64 & seed_64);
			if(i >= (seed_len - 1)){
				num_insertion ++;
				insert_smer_to_ht(temp_smer, i-seed_len+1, it->first);
			}
		}
	}
}
void HASH_TABLE :: insert_smer_to_ht(uint64_t new_smer, int sta, int pro_id){
	uint64_t temp_index = get_ht_index(new_smer % ht_size, new_smer);
	hash_table[temp_index].smer = new_smer;
	smer_detail temp_occ;
	temp_occ.pro_id = pro_id;
	temp_occ.sta_pos = sta;
	hash_table[temp_index].smer_occs.push_back(temp_occ);
}
uint64_t HASH_TABLE :: get_ht_index(uint64_t ind, uint64_t smer){
	if(hash_table[ind].smer == hash_table_default) return ind;
	if(hash_table[ind].smer == smer) return ind;
	else return get_ht_index((ind + 1) % ht_size, smer);
}

uint64_t HASH_TABLE :: search_in_ht(uint64_t index, uint64_t target_smer){
	if(hash_table[index].smer == hash_table_default){
		return hash_table_default;
	}
	else if(hash_table[index].smer == target_smer){
		return index;
	}
	else{
		return search_in_ht((index + 1)%ht_size, target_smer);
	}
}

#endif /* HASH_TABLE_H_ */
