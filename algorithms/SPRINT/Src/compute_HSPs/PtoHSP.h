#ifndef PTOHSP_H_
#define PTOHSP_H_
#include "global_parameters.h"
#include "hash_table.h"

class HSP_pair{	//this is stored in HSP_PAIRS
public:
	int p1_sta;
	int p2_sta;
	int length;
	HSP_pair(int a, int b, int c){
		p1_sta = a;
		p2_sta = b;
		length = c;
	}
	bool operator < (const HSP_pair &a2) const {
		  if (p1_sta < a2.p1_sta) return true;
		  else if((p1_sta == a2.p1_sta) && (p2_sta < a2.p2_sta)) return true;
		  else return false;
	}
};

class PtoHSP{	//this class aims to compute all HSP and store them in HSP_PAIRS
public:
	boost::unordered_map < pair <int, int>, set <HSP_pair> > HSP_PAIRS;
	#ifdef PARAL
	omp_lock_t lock;
	#endif
	PtoHSP();
	void load_hash_table(HASH_TABLE & ht);	//basaed on a hash_table, investigate hit, extend, mark HSP
	void compute_similar_smers(uint64_t smer1, HASH_TABLE & ht, set <uint64_t> & similar_smers);	//compute H(x)
	void investigate_hits(uint64_t smer1_index, uint64_t smer2_index, HASH_TABLE & ht);	//check if this is a hit
	void change_smer_digit(uint64_t smer1, int digit_pos, HASH_TABLE & ht, int score_needed, set <uint64_t> & similar_smers); //change one digit one a smer. e.g. AND0RB could be changed as AND0RQ for digit 0
	uint64_t get_smer_dig_plus1(uint64_t smer1, int digit_pos, uint64_t seed);
	void put_window(int p1_id, int p2_id, int sta1, int sta2, HASH_TABLE & ht);//put the k_size window on top
	void extend_HSP(int p1_id, int p2_id, int sta1, int sta2, HASH_TABLE & ht, int score);//extend HSP to both left and right side
	void print_HSP();
	void load_identi_sub_seq(string sub_seq_fn);
	void extend_and_store_this_sub_seq_as_hsp(int p1_id, int p2_id, int sta1, int sta2, int score);
};

void PtoHSP :: print_HSP(){
	if(ONLY_COMPUTE_NEW_PROTEIN){	//for each protein, add a hsp(0,lenth_of_pro), but only for newly added proteins, that is, starting from id num_old_protein
		for(int a = num_old_protein; a < num_protein; a ++){
			HSP_pair p(0, 0, p_id_seq.at(a).length());
			// HSP_PAIRS.insert(make_pair(make_pair(a, a), p));
			HSP_PAIRS[make_pair(a, a)].insert(p);
		}	
	}
	else{
		//for each protein, add a hsp(0,lenth_of_pro) 
		for(int a = 0; a < num_protein; a ++){
			HSP_pair p(0, 0, p_id_seq.at(a).length());
			// HSP_PAIRS.insert(make_pair(make_pair(a, a), p));
			HSP_PAIRS[make_pair(a, a)].insert(p);
		}		
	}

	//output HSP_PAIRS
	cout<<"----------output the HSP_PAIRS-------"<<endl;
	ofstream fout;
	if(ONLY_COMPUTE_NEW_PROTEIN){
		fout.open (ORIGINAL_HSP_FN.c_str(), std::ofstream::app);//append to orginal hsp file
	}
	else{
		fout.open (HSP_FN.c_str(), std::ofstream::out);//create new hsp file
	}
	for(boost::unordered_map < pair <int, int>, set <HSP_pair> >:: iterator it = HSP_PAIRS.begin(); it != HSP_PAIRS.end(); it++){
		//indicating which two proteins
		fout<<"> "<<p_id_name.at(it->first.first)<<" and "<<p_id_name.at(it->first.second)<<"\n";
		for(set <HSP_pair>::iterator it2 = (it->second.begin()); it2 != (it->second.end()); it2 ++){
			fout<<it2->p1_sta<<" "<<it2->p2_sta<<" "<<it2->length<<"\n";
			// fout<<it2->p1_sta<<" "<<it2->p2_sta<<" "<<it2->length<<" "<<compare_two_strings(it->first.first,it->first.second,it2->p1_sta,it2->p2_sta,it2->length)<<"\n";
		}
	}
	fout.close();
}
PtoHSP :: PtoHSP(){
	//initialize lock
	#ifdef PARAL
	omp_init_lock(&(lock));
	#endif
}
void PtoHSP :: load_hash_table(HASH_TABLE & ht){
	#ifdef PARAL
	#pragma omp parallel for schedule(dynamic)	
	#endif
		for(uint64_t a = 0; a < ht.ht_size; a ++){
			if(STAS && (a % 15000 == 0)){
				#ifdef PARAL
				#pragma omp critical(writeFile)
				#endif
				cout<<a <<"/ "<<ht.ht_size<<" done"<<endl;
			}
			set <uint64_t> similar_smers;
			if(ht.hash_table[a].smer != hash_table_default){
				compute_similar_smers(ht.hash_table[a].smer, ht, similar_smers);
				for(set <uint64_t>::iterator it = similar_smers.begin(); it != similar_smers.end(); it ++){
					uint64_t smer2_index = ht.search_in_ht((*it) % ht.ht_size, (*it));
					if((smer2_index != hash_table_default) && (smer2_index >= a)){//only compare a with the entries that below it, so that we avoid duplicated investigation
						investigate_hits(a, smer2_index, ht);
					}
				}
				similar_smers.clear();
			}
		}		
}
void PtoHSP :: compute_similar_smers(uint64_t smer1, HASH_TABLE & ht, set <uint64_t> & similar_smers){
	int smer_score = 0; //score(smer1, smer1)
	uint64_t seed_current_digit = 0;
	uint64_t smer_current_digit = 0;
	for(uint32_t a = 1; a <  ht.seed_len; a ++){ //starting from 1, because digit0 is not considered here
		seed_current_digit = ht.seed_64 & ((uint64_t)31 << (a * 5));
		if(seed_current_digit){
			smer_current_digit = smer1 & ((uint64_t)31 << (a * 5));
			smer_current_digit = smer_current_digit >> (a * 5);
			smer_score += BLOSUM80[smer_current_digit][smer_current_digit];
		}
	}
	change_smer_digit(smer1, 0, ht, Thit - smer_score, similar_smers);
}
void PtoHSP :: change_smer_digit(uint64_t smer1, int digit_pos, HASH_TABLE & ht, int score_needed, set <uint64_t> & similar_smers){
	if(digit_pos == (int)(ht.seed_len)) return;
	uint64_t seed_current_digit = (ht.seed_64 & ((uint64_t)31 <<(digit_pos * 5)));
	if(seed_current_digit == 0){	// the 0 position in seed, just add 0, no need to change anything
		change_smer_digit(smer1, digit_pos + 1, ht, score_needed, similar_smers);
	}
	else if(digit_pos == (int) (ht.seed_len) - 1){	//the last digit to change
		uint64_t smer_cunt_dig = (smer1 & ((uint64_t)31 << (digit_pos * 5))) >> (digit_pos * 5); //smer's current digit value
		uint64_t y;
		for(int j = 0; j < 20; j ++){
			y = 0;
			if(BLOSUM80[smer_cunt_dig][B80_ordered[smer_cunt_dig][j]] >= score_needed){
				y = smer1 & ((uint64_t)~((uint64_t)31 << digit_pos * 5));//  1. y's current_digit = 00000(y & ~111011), 2. y's current_digit = j(y | 00001000)
				y = y | ((uint64_t)(B80_ordered[smer_cunt_dig][j]) << (5 * digit_pos));
				similar_smers.insert(y);
			}
			else break;
		}
	}
	else{
		uint64_t smer_cunt_dig = (smer1 & ((uint64_t)31 << (digit_pos * 5))) >> (digit_pos * 5);
		uint64_t smer_cunt_dig_plus1 = get_smer_dig_plus1(smer1, digit_pos, ht.seed_64);
		uint64_t y;
		int new_score_needed = 0;
		for(int j = 0; j < 20; j ++){
			y = 0;
			if(BLOSUM80[smer_cunt_dig][B80_ordered[smer_cunt_dig][j]] >= score_needed){
				y = smer1 & ((uint64_t)~((uint64_t)31 << digit_pos * 5));
				y = y | ((uint64_t)(B80_ordered[smer_cunt_dig][j]) << (5 * digit_pos));
				similar_smers.insert(y);
				new_score_needed = score_needed + BLOSUM80[smer_cunt_dig_plus1][smer_cunt_dig_plus1] - BLOSUM80[smer_cunt_dig][B80_ordered[smer_cunt_dig][j]];
				change_smer_digit(y, digit_pos + 1, ht, new_score_needed, similar_smers);
			}
			else break;
		}
	}
}
uint64_t PtoHSP :: get_smer_dig_plus1(uint64_t smer1, int digit_pos, uint64_t seed){
	uint64_t seed_current_digit = seed & ((uint64_t)31 << ((digit_pos+1) * 5));
	if(seed_current_digit){
		 return (smer1 & ((uint64_t)31 << ((digit_pos+1) * 5))  ) >> ((digit_pos + 1) * 5);
	}
	else{
		return get_smer_dig_plus1(smer1, digit_pos + 1, seed);
	}

}
void PtoHSP :: investigate_hits(uint64_t smer1_index, uint64_t smer2_index, HASH_TABLE & ht){
	int smer1_cnt = ht.hash_table[smer1_index].smer_occs.size();
	int smer2_cnt = ht.hash_table[smer2_index].smer_occs.size();
	int p1_id, p2_id, sta1, sta2;
	// boost::unordered_map < pair <int, int>, set <HSP_pair> > :: iterator itr1;
	int NEED_CHECK = 1; //if the hits is within a detected HSP, set it to 0

	if(ht.hash_table[smer1_index].smer == ht.hash_table[smer2_index].smer){	//same smer, compare without diagonal
		for(int a = 0; a < smer1_cnt; a ++){
			for(int b = a + 1; b < smer2_cnt; b ++){
				NEED_CHECK = 1;
				p1_id = ht.hash_table[smer1_index].smer_occs[a].pro_id;
				p2_id = ht.hash_table[smer2_index].smer_occs[b].pro_id;

				if(ONLY_COMPUTE_NEW_PROTEIN){//check if the proteins are new proteins: if old, skip
					if((new_proteins.find(p1_id) != new_proteins.end()) ||  (new_proteins.find(p2_id) != new_proteins.end()) ){//at least one of them needs to be new
						NEED_CHECK = 1;
					}
					else{
						NEED_CHECK = 0;
					}
				}
				if(NEED_CHECK){
					sta1 = ht.hash_table[smer1_index].smer_occs[a].sta_pos;
					sta2 = ht.hash_table[smer2_index].smer_occs[b].sta_pos;
					if(p1_id > p2_id){
						swap(p1_id, p2_id);
						swap(sta1, sta2);
					}				
					if((ht.hash_table[smer1_index].smer_occs[a].pro_id != ht.hash_table[smer2_index].smer_occs[b].pro_id) || (ht.hash_table[smer1_index].smer_occs[a].sta_pos != ht.hash_table[smer2_index].smer_occs[b].sta_pos) ){
						// #ifdef PARAL
						// omp_set_lock(&lock);
						// #endif
						// itr1 = HSP_PAIRS.find(make_pair(p1_id, p2_id));
						// #ifdef PARAL
						// omp_unset_lock(&lock);
						// #endif
						// if(itr1 != HSP_PAIRS.end()){
						// 	for(set <HSP_pair>  :: iterator itr2 = itr1->second.begin(); itr2 != itr1->second.end(); itr2 ++){
						// 		if((sta1 >= itr2->p1_sta) && (sta2 >= itr2->p2_sta) && (sta1 + (int)ht.seed_len <= itr2->p1_sta + itr2->length) && (sta2 + (int)ht.seed_len <= itr2->p2_sta + itr2->length) && (sta1 - itr2->p1_sta == sta2 - itr2->p2_sta)){
						// 			NEED_CHECK = 0;
						// 			break;
						// 		}
						// 	}
						// }		
						if(NEED_CHECK){			
							put_window(p1_id, p2_id, sta1, sta2, ht);
						}
					}
				}
			}
		}
	}
	else{	//different smers, compare with diagonal
		for(int a = 0; a < smer1_cnt; a ++){
			for(int b = 0; b < smer2_cnt; b ++){
				NEED_CHECK = 1;
				p1_id = ht.hash_table[smer1_index].smer_occs[a].pro_id;
				p2_id = ht.hash_table[smer2_index].smer_occs[b].pro_id;
				if(ONLY_COMPUTE_NEW_PROTEIN){//check if the proteins are new proteins: if old, skip
					if((new_proteins.find(p1_id) != new_proteins.end()) ||  (new_proteins.find(p2_id) != new_proteins.end()) ){//at least one of them needs to be new
						NEED_CHECK = 1;
					}
					else{
						NEED_CHECK = 0;
					}
				}	
				if(NEED_CHECK){			
					sta1 = ht.hash_table[smer1_index].smer_occs[a].sta_pos;
					sta2 = ht.hash_table[smer2_index].smer_occs[b].sta_pos;		
					if(p1_id > p2_id){
						swap(p1_id, p2_id);
						swap(sta1, sta2);
					}		
					// #ifdef PARAL
					// omp_set_lock(&lock);
					// #endif
					// itr1 = HSP_PAIRS.find(make_pair(p1_id, p2_id));
					// #ifdef PARAL
					// omp_unset_lock(&lock);
					// #endif
					// if(itr1 != HSP_PAIRS.end()){
					// 	for(set <HSP_pair>  :: iterator itr2 = itr1->second.begin(); itr2 != itr1->second.end(); itr2 ++){
					// 		if((sta1 >= itr2->p1_sta) && (sta2 >= itr2->p2_sta) && (sta1 + (int)ht.seed_len <= itr2->p1_sta + itr2->length) && (sta2 + (int)ht.seed_len <= itr2->p2_sta + itr2->length) && (sta1 - itr2->p1_sta == sta2 - itr2->p2_sta)){
					// 			NEED_CHECK = 0;
					// 			break;
					// 		}
					// 	}
					// }
					if(NEED_CHECK){
						put_window(p1_id, p2_id, sta1, sta2, ht);
					}
				}
			}
		}
	}
}

void PtoHSP :: put_window(int p1_id, int p2_id, int sta1, int sta2, HASH_TABLE & ht){	//considering put the k-sized window for every possible position in the future
	int num_check = k_size - (int)ht.seed_len + 1;
	int score = 0;
	for(int pos = 0; pos < num_check; pos ++){
		if((sta1 - pos >= 0) && (sta2 - pos >= 0) && (sta1 - pos + k_size <= (int)(p_id_seq.at(p1_id).length())) && (sta2 - pos + k_size <= (int) (p_id_seq.at(p2_id).length()) )){
			score = compare_two_strings(p1_id, p2_id, sta1 - pos, sta2 - pos, k_size);
			if(score >= T_kmer){
				extend_HSP(p1_id, p2_id, sta1 - pos, sta2 - pos, ht, score);
				break;
			} 
		}
	}

}

void PtoHSP :: extend_HSP(int p1_id, int p2_id, int sta1, int sta2, HASH_TABLE & ht, int score){
	//extend to the right
		int new_sta1 = sta1;
		int new_sta2 = sta2;
		int new_end1 = sta1 + k_size - 1;
		int new_end2 = sta2 + k_size - 1;
		int pos_1l = sta1;	//pointer to the current char in protein1 left side
		int pos_2l = sta2;
		int pos_1r = new_end1;		//pointer to the current char in protein1 right side
		int pos_2r = new_end2;
		int to_right;
		int to_left;
		int current_score = score;
		if(p_id_seq[p1_id].length() - sta1 > p_id_seq[p2_id].length() - sta2){
			to_right = p_id_seq[p2_id].length() - (sta2 + k_size);
		}
		else{
			to_right = p_id_seq[p1_id].length() - (sta1 + k_size);
		}

		for(int a = 0; a < to_right; a ++){
			current_score = current_score - BLOSUM_score(p_id_seq[p1_id].at(pos_1l),p_id_seq[p2_id].at(pos_2l)) + BLOSUM_score(p_id_seq[p1_id].at(pos_1r+1),p_id_seq[p2_id].at(pos_2r+1));
			if(current_score >= T_kmer){
				pos_1l ++;
				pos_2l ++;
				pos_1r ++;
				pos_2r ++;
				if(a == to_right - 1){
					new_end1 = pos_1r;
					new_end2 = pos_2r;
				}
			}
			else{
				new_end1 = pos_1r;
				new_end2 = pos_2r;
				break;
			}
		}
		pos_1l = sta1;
		pos_2l = sta2;
		pos_1r = sta1 + k_size - 1;
		pos_2r = sta2 + k_size - 1;
		current_score = score;
	//extend to the left
		if(sta1 > sta2){
			to_left = sta2;
		}
		else{
			to_left = sta1;
		}
		for(int a = 0; a < to_left; a ++){
			current_score = current_score - BLOSUM_score(p_id_seq[p1_id].at(pos_1r),p_id_seq[p2_id].at(pos_2r)) + BLOSUM_score(p_id_seq[p1_id].at(pos_1l-1),p_id_seq[p2_id].at(pos_2l-1));
			if(current_score >= T_kmer){
				pos_1l --;
				pos_2l --;
				pos_1r --;
				pos_2r --;
				if(a == to_left - 1){
					new_sta1 = pos_1l;
					new_sta2 = pos_2l;
				}
			}
			else{
				new_sta1 = pos_1l;
				new_sta2 = pos_2l;
				break;
			}
		}

		//no need to check p1, p2 here, because after the function "put_window" all p1_id<p2_id here
		HSP_pair p(new_sta1, new_sta2, new_end1 - new_sta1 + 1);
		#ifdef PARAL
		omp_set_lock(&lock);	
		#endif
		HSP_PAIRS[make_pair(p1_id,p2_id)].insert(p);
		#ifdef PARAL
		omp_unset_lock(&lock);
		#endif
}
void PtoHSP :: load_identi_sub_seq(string sub_seq_fn){
	ifstream fin(sub_seq_fn.c_str());
	if(!fin)
	{
		cout<<"error opening file "<<sub_seq_fn<<endl;
		exit(3);
	}
	
	string temp1, temp2, temp3, temp4;
	int score = 0; //score between two 20-mer sub-seqs
	int temp_p1_id = -1, temp_p2_id = -1;
	int PRTEIN_FOUND = 0;//0: protein name in the HSP is not in the sequence file; 1: otherwise
	while(fin >> temp1 >> temp2 >> temp3){
		if(temp1 == ">"){	//read protein info, > p1 and p2
			PRTEIN_FOUND = 0;
			fin >> temp4;
			// temp1: >; temp2: p1_name; temp3: and; temp4: p2_name
			if((p_name_id.find(temp2) != p_name_id.end()) && (p_name_id.find(temp4) != p_name_id.end())){
				PRTEIN_FOUND = 1;
				temp_p1_id = p_name_id.at(temp2);
				temp_p2_id = p_name_id.at(temp4);
			}
			else{
				if(p_name_id.find(temp2) == p_name_id.end()) cout<<"In the HSP file, Protein "<<temp2<<" could not be found in the protein sequence file.\n";
				if(p_name_id.find(temp4) == p_name_id.end()) cout<<"In the HSP file, Protein "<<temp4<<" could not be found in the protein sequence file.\n";
			}
		}
		else{//read hsp info if the proteins are in the protein sequences file
			if(PRTEIN_FOUND){
				//temp1: sta1; temp2: sta2; temp3: length
				score = compare_two_strings(temp_p1_id, temp_p2_id, atoi(temp1.c_str()), atoi(temp2.c_str()), k_size);
				extend_and_store_this_sub_seq_as_hsp(temp_p1_id, temp_p2_id, atoi(temp1.c_str()), atoi(temp2.c_str()), score);	
			}
		}
	}
	fin.close();	
}

void PtoHSP :: extend_and_store_this_sub_seq_as_hsp(int p1_id, int p2_id, int sta1, int sta2, int score){
		if(p1_id > p2_id){	//make sure p1_id < p2_id
			swap(p1_id, p2_id);
			swap(sta1, sta2);
		}
		//extend to the right
		int new_sta1 = sta1;
		int new_sta2 = sta2;
		int new_end1 = sta1 + k_size - 1;
		int new_end2 = sta2 + k_size - 1;
		int pos_1l = sta1;	//pointer to the current char in protein1 left side
		int pos_2l = sta2;
		int pos_1r = new_end1;		//pointer to the current char in protein1 right side
		int pos_2r = new_end2;
		int to_right;
		int to_left;
		int current_score = score;
		if(p_id_seq[p1_id].length() - sta1 > p_id_seq[p2_id].length() - sta2){
			to_right = p_id_seq[p2_id].length() - (sta2 + k_size);
		}
		else{
			to_right = p_id_seq[p1_id].length() - (sta1 + k_size);
		}

		for(int a = 0; a < to_right; a ++){
			current_score = current_score - BLOSUM_score(p_id_seq[p1_id].at(pos_1l),p_id_seq[p2_id].at(pos_2l)) + BLOSUM_score(p_id_seq[p1_id].at(pos_1r+1),p_id_seq[p2_id].at(pos_2r+1));
			if(current_score >= T_kmer){
				pos_1l ++;
				pos_2l ++;
				pos_1r ++;
				pos_2r ++;
				if(a == to_right - 1){
					new_end1 = pos_1r;
					new_end2 = pos_2r;
				}
			}
			else{
				new_end1 = pos_1r;
				new_end2 = pos_2r;
				break;
			}
		}
		pos_1l = sta1;
		pos_2l = sta2;
		pos_1r = sta1 + k_size - 1;
		pos_2r = sta2 + k_size - 1;
		current_score = score;
	//extend to the left
		if(sta1 > sta2){
			to_left = sta2;
		}
		else{
			to_left = sta1;
		}
		for(int a = 0; a < to_left; a ++){
			current_score = current_score - BLOSUM_score(p_id_seq[p1_id].at(pos_1r),p_id_seq[p2_id].at(pos_2r)) + BLOSUM_score(p_id_seq[p1_id].at(pos_1l-1),p_id_seq[p2_id].at(pos_2l-1));
			if(current_score >= T_kmer){
				pos_1l --;
				pos_2l --;
				pos_1r --;
				pos_2r --;
				if(a == to_left - 1){
					new_sta1 = pos_1l;
					new_sta2 = pos_2l;
				}
			}
			else{
				new_sta1 = pos_1l;
				new_sta2 = pos_2l;
				break;
			}
		}

		//no need to check p1, p2 here, because after the function "put_window" all p1_id<p2_id here
		HSP_pair p(new_sta1, new_sta2, new_end1 - new_sta1 + 1);
		HSP_PAIRS[make_pair(p1_id,p2_id)].insert(p);
}
#endif /* PTOHSP_H_ */
