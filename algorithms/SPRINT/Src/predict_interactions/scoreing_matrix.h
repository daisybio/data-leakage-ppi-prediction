#ifndef SCOREING_MATRIX_H_
#define SCOREING_MATRIX_H_
#include "global_parameters.h"
#include "PtoHSP.h"

typedef std::pair<int, int> pair_type; 

class int_pair{	//interaction pairs. Each entry is a pair record.
public:
	int p1_id;
	int p2_id;
	float score;
	char flag;	//'n': neg; 'p': pos
	int_pair(int a, int b, float c, char d){
		p1_id = a;
		p2_id = b;
		score = c;
		flag = d;
	}
   bool operator < (const int_pair &a2) const {
	  return (score > a2.score);
   }
};
class SCORING_MATRIX{
public:
	int num_pos, num_neg;
	float ** final_score_matrix;//2D matrix, each cell is the score
	vector <pair<int, int> > training_set;//store all training pairs
	SCORING_MATRIX(PtoHSP & hsp);
	void load_traing(string traing_file_name, PtoHSP & hsp);	//reading from training set, and adding the score into SCORE_MAT
	void load_test(string pos_file_name, char flag);//could load both pos_set or neg_set, char flag; 'n': neg; 'p': pos
	void print_entire_final_score_matrix();
};
SCORING_MATRIX::SCORING_MATRIX(PtoHSP & hsp){
	num_pos = 0;
	num_neg = 0;
	final_score_matrix = (float **) malloc(sizeof(float *) * num_protein);
    for(int a = 0; a < num_protein; a ++)
    {
    	final_score_matrix[a] = (float *) malloc(sizeof(float) * (num_protein));
    }
    for(int a = 0; a < num_protein; a ++)
    {
    	for(int b = 0; b < num_protein; b ++){
    		final_score_matrix[a][b] = 0.0;
    	}
    }
}
void SCORING_MATRIX :: load_traing(string traing_file_name, PtoHSP & hsp){

	ifstream fin(traing_file_name.c_str());
	if(!fin)
	{
		cout<<"error opening file "<<traing_file_name<<endl;
		exit(3);
	}
	string temp1, temp2;
	int temp_p1_id, temp_p2_id;
	int staticstic = 0;
	int num_pairs = 0;
	
	cout<<"reading training file: "<<traing_file_name<<endl;
	while(fin >> temp1 >> temp2){
		num_pairs ++;
		if((p_name_id.find(temp1) != p_name_id.end()) && (p_name_id.find(temp2) != p_name_id.end())){
			temp_p1_id = p_name_id[temp1];
			temp_p2_id = p_name_id[temp2];
			if(temp_p2_id >= temp_p1_id){
				swap(temp_p1_id, temp_p2_id);
			}
			training_set.push_back(make_pair(temp_p1_id, temp_p2_id));
		}
	}
	fin.close();
	#ifdef PARAL
	#pragma omp parallel for schedule(dynamic)
	#endif
	for(int a = 0; a < (int)training_set.size(); a ++){
		if(STATS){
			staticstic ++;
			if(staticstic % 100 == 0){
				cout<<"now reading the "<<staticstic<<" training pair"<<endl;
			}			
		}
		int p1_id, p2_id, p1_sim_id, p2_sim_id;
		p1_id = training_set.at(a).first;
		p2_id = training_set.at(a).second;
		if(p1_id != p2_id){
			for(int i = 0; i < (int)hsp.HSP_table.at(p1_id).size(); i ++){
				for(int j = 0; j < (int)hsp.HSP_table.at(p2_id).size(); j ++){
					p1_sim_id = hsp.HSP_table.at(p1_id).at(i).p2_id;
					p2_sim_id = hsp.HSP_table.at(p2_id).at(j).p2_id;
					//final_score_matrix += (score_hsp1 * (len2-19) + score_hsp2 * (len1-19))
					final_score_matrix[p1_sim_id][p2_sim_id] += (hsp.HSP_table.at(p1_id).at(i).hsp_calculation_score * (hsp.HSP_table.at(p2_id).at(j).length- k_size + 1) + hsp.HSP_table.at(p2_id).at(j).hsp_calculation_score * (hsp.HSP_table.at(p1_id).at(i).length- k_size + 1));
					final_score_matrix[p2_sim_id][p1_sim_id] = final_score_matrix[p1_sim_id][p2_sim_id];
				}
			}
		}
		else{// trainig pair which contains same protein, only do half matrix, including the diagonal line
			for(int i = 0; i < (int)hsp.HSP_table.at(p1_id).size(); i ++){
				for(int j = i; j < (int)hsp.HSP_table.at(p2_id).size(); j ++){
					p1_sim_id = hsp.HSP_table.at(p1_id).at(i).p2_id;
					p2_sim_id = hsp.HSP_table.at(p2_id).at(j).p2_id;
					//final_score_matrix += (score_hsp1 * (len2-19) + score_hsp2 * (len1-19))
					final_score_matrix[p1_sim_id][p2_sim_id] += (hsp.HSP_table.at(p1_id).at(i).hsp_calculation_score * (hsp.HSP_table.at(p2_id).at(j).length- k_size + 1) + hsp.HSP_table.at(p2_id).at(j).hsp_calculation_score * (hsp.HSP_table.at(p1_id).at(i).length- k_size + 1));
					final_score_matrix[p2_sim_id][p1_sim_id] = final_score_matrix[p1_sim_id][p2_sim_id];
				}
			}
		}

	}
	cout<<"finish reading this training set. The size of the training set is: "<<num_pairs<<" pairs."<<endl;
	cout<<"compute the volum_sum for each pair"<<endl;
	#ifdef PARAL
	#pragma omp parallel for schedule(dynamic)
	#endif
	for(int a = 0; a < num_protein; a ++){
		for(int b = a; b < num_protein; b ++){
			final_score_matrix[a][b] = final_score_matrix[a][b] / ( (float)p_id_seq.at(a).length() * (float)p_id_seq.at(b).length() );
			final_score_matrix[b][a] = final_score_matrix[a][b];
		}
	}	
}

void SCORING_MATRIX :: load_test(string pos_file_name, char flag){
	ofstream fout;
	fout.open(OUTPUT_FN.c_str(), ios::app);
	ofstream fout_pos;
	fout_pos.open(OUTPUT_pos_FN.c_str(), ios::app);
	ofstream fout_neg;
	fout_neg.open(OUTPUT_neg_FN.c_str(), ios::app);

	ifstream fin(pos_file_name.c_str());
	if(!fin)
	{
		cout<<"error opening file "<<pos_file_name<<endl;
		exit(3);
	}
	cout<<"reading testing file: "<<pos_file_name<<endl;
	string temp1, temp2;
	int temp_p1id, temp_p2id;
	while(fin >> temp1 >> temp2){
		if((p_name_id.find(temp1) != p_name_id.end()) && (p_name_id.find(temp2) != p_name_id.end())){
			if(flag == 'p') {num_pos ++;}
			else {num_neg ++;}
			temp_p1id = p_name_id.at(temp1);
			temp_p2id = p_name_id.at(temp2);
			if(flag == 'p'){
				fout<<final_score_matrix[temp_p1id][temp_p2id]<<" 1\n";
				fout_pos<<final_score_matrix[temp_p1id][temp_p2id]<<" 1\n";
			}
			else{
				fout<<final_score_matrix[temp_p1id][temp_p2id]<<" 0\n";
				fout_neg<<final_score_matrix[temp_p1id][temp_p2id]<<" 0\n";
			}
		}
	}
	fin.close();
	fout.close();
	fout_pos.close();
	fout_neg.close();
}
void SCORING_MATRIX :: print_entire_final_score_matrix(){
	ofstream fout(OUTPUT_FN.c_str());
	for(int a = 0; a < num_protein; a ++){
		for(int b = a; b < num_protein; b ++){
			fout<<p_id_name.at(a)<<" "<<p_id_name.at(b)<<" "<<final_score_matrix[a][b]<<"\n";
		}
	}
	fout.close();
}

#endif /* SCOREING_MATRIX_H_ */
