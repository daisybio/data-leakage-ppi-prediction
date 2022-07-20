#ifndef PTOHSP_H_
#define PTOHSP_H_
#include "global_parameters.h"


class HSP_OCC{	//a single HSP occurence, this is stored in HSP_table
public:
	int p2_id;	//protein2 id
	int p1_sta;	//protein1's starting position
	int p2_sta;	//protein2's starting position
	int length;	//the length of this HSP occurence
	float hsp_calculation_score; //score with overlap
	HSP_OCC(int a, int b, int c, int d, float e){
		p2_id = a;
		p1_sta = b;
		p2_sta = c;
		length = d;
		hsp_calculation_score = e;
	}
};
class HSP_entry{	//a whole entry of a protein, it contaions all HSP it involves in. The index is its protein id
public:
	vector <HSP_OCC> hsp_occs;
};
class PtoHSP{	//this PtoHSP is different from the one in the output sprint. This class reads from the output HSP pairs, and store them in HSP_table
public:
//	array <vector <HSP_OCC>, 7000 > HSP_table;//warning here, I couldn't find a way to pass num_protein to this initialization, the only way I can do here is to use 7000, remember to change it when testing on other organism
//	HSP_entry * HSP_table;//the index is p1_id, the rest is hsp_occ which is constists of p2_id, p1_sta, p2_sta, length
	vector <vector <HSP_OCC> > HSP_table;
	PtoHSP();
	void load_HSP_pairs();
	void print_HSP_table();
	void sort_HSP_table();
	void mark_flag(int p1id, int sta1, int len);
	void pre_load_HSP_pairs();
	void store_this_hsp(int p1, int p2, int sta1, int sta2, int hsp_len);
};
bool myComp(const HSP_OCC & first, const HSP_OCC & second){
	return (first.p2_id < second.p2_id);
}
PtoHSP :: PtoHSP(){
	cout<<"num_protein: "<<num_protein<<endl;
	HSP_table.reserve(num_protein);
	for(int a = 0; a < num_protein; a ++){
		HSP_table.push_back(vector <HSP_OCC>());
	}
	pre_load_HSP_pairs();
	cout<<"Processing HSP pairs..."<<endl;
	load_HSP_pairs();
	sort_HSP_table();
}
void PtoHSP :: load_HSP_pairs(){
	ifstream fin(HSP_FN.c_str());
	if(!fin)
	{
		cout<<"error opening file "<<HSP_FN<<endl;
		exit(3);
	}
	cout<<"loading HSP file: "<<HSP_FN<<endl;
	string temp1, temp2, temp3, temp4;
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
				if ((temp_p1_id == temp_p2_id) && ( atoi(temp3.c_str()) == (int)(p_id_seq.at(temp_p1_id).length()))) { //hsp with itsefl
					HSP_OCC temp_hsp(temp_p2_id, atoi(temp1.c_str()), atoi(temp2.c_str()), atoi(temp3.c_str()), (float)socre_between_two_hsp(temp_p1_id, temp_p2_id, 0, 0, atoi(temp3.c_str()))  );
					HSP_table.at(temp_p1_id).push_back(temp_hsp);					
				}
				else{	//normal HSPs
					store_this_hsp(temp_p1_id, temp_p2_id, atoi(temp1.c_str()), atoi(temp2.c_str()), atoi(temp3.c_str()));
				}				
			}
		}
	}
	fin.close();
}
void PtoHSP :: print_HSP_table(){
	for(int a = 0; a < num_protein; a ++){
		if(HSP_table.at(a).size() != 0){
			cout<<"---"<<endl;
			for(uint32_t b = 0; b < HSP_table.at(a).size(); b ++){
				cout<<a<<" "<<HSP_table.at(a).at(b).p2_id<<" "<<HSP_table.at(a).at(b).p1_sta<<" "<<HSP_table.at(a).at(b).p2_sta<<" "<<HSP_table.at(a).at(b).length<<endl;
			}
		}
	}
}
void PtoHSP :: sort_HSP_table(){
	for(int a = 0; a < num_protein; a ++){
		sort(HSP_table.at(a).begin(), HSP_table.at(a).end(), myComp);
	}
}
void PtoHSP :: mark_flag(int p1id, int sta1, int len) {
	for (int a = 0; a < len - 19; a++) {
		hsp_flag.at(p1id).at(a + sta1) ++;
	}
}
void PtoHSP :: pre_load_HSP_pairs(){
	ifstream fin(HSP_FN.c_str());
	if (!fin) {
		cout << "error opening file " << HSP_FN << endl;
		exit(3);
	}
	string temp1, temp2, temp3;
	int temp_p1_id = -1, temp_p2_id = -1;
	int PROTEIN_FOUND = 1;
	while (fin >> temp1 >> temp2 >> temp3) {
		if (temp1 == ">") {
			PROTEIN_FOUND = 1;
			//temp1: >; temp2: p1name; temp3: andm
			if(p_name_id.find(temp2) != p_name_id.end()){
				temp_p1_id = p_name_id.at(temp2);
				fin >> temp1;
				//temp1: p2name
				if(p_name_id.find(temp1) != p_name_id.end()){
					temp_p2_id = p_name_id.at(temp1);
				}
				else{
					cout<<"warning: protein "<<temp1<<" in the HSP file could not be found in the protein sequence file."<<endl;
					PROTEIN_FOUND = 0;
				}				
			}
			else{
				cout<<"warning: protein "<<temp2<<" in the HSP file could not be found in the protein sequence file."<<endl;
				PROTEIN_FOUND = 0;
			}
		} else {
			if(PROTEIN_FOUND){
				mark_flag(temp_p1_id, atoi(temp1.c_str()), atoi(temp3.c_str()));
				mark_flag(temp_p2_id, atoi(temp2.c_str()), atoi(temp3.c_str()));
			}
		}
	}
	fin.close();

}
void PtoHSP :: store_this_hsp(int p1, int p2, int sta1, int sta2, int hsp_len){
	for(int a = 0; a < hsp_len - 19; a ++){
		if(a == hsp_len - 20){ //last position
			if((hsp_flag.at(p1).at(sta1 + a) <= T_hsp_max) && (hsp_flag.at(p2).at(sta2 + a) <= T_hsp_max )){ //last position lower than T_hsp_max
				HSP_OCC temp_hsp(p2, sta1, sta2, hsp_len, (float)socre_between_two_hsp(p1, p2, sta1, sta2, hsp_len)  );
				HSP_table.at(p1).push_back(temp_hsp);
				if(p1 != p2){
					HSP_OCC temp_hsp2(p1, sta2, sta1, hsp_len, (float)socre_between_two_hsp(p1, p2, sta1, sta2, hsp_len));
					HSP_table.at(p2).push_back(temp_hsp2);
				}
				break;
			}		
		}
		if((hsp_flag.at(p1).at(sta1 + a) > T_hsp_max) || (hsp_flag.at(p2).at(sta2 + a) > T_hsp_max)){
			if(a){
				//temp1: sta1; temp2: sta2; temp3: length
				HSP_OCC temp_hsp(p2, sta1, sta2, 19 + a, (float)socre_between_two_hsp(p1, p2, sta1, sta2, 19 + a)  );
				HSP_table.at(p1).push_back(temp_hsp);
				if(p1 != p2){
					HSP_OCC temp_hsp2(p1, sta2, sta1, 19 + a, (float)socre_between_two_hsp(p1, p2, sta1, sta2, 19 + a) );
					HSP_table.at(p2).push_back(temp_hsp2);
				}
			}
			if(hsp_len - a >= 21){
				store_this_hsp(p1, p2, sta1 + a + 1, sta2 + a + 1, hsp_len - a -1);
				break;
			}
		}
	}
}
#endif /* PTOHSP_H_ */
