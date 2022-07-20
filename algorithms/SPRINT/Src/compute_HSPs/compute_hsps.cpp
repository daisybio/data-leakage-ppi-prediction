//============================================================================
// Name        : compute_hsps.cpp
// Author      : Yiwei Li
// Date		   : Date: May 2016
//============================================================================
#include "global_parameters.h"
#include "hash_table.h"
#include "PtoHSP.h"

int main(int argc, char * argv[]) {
	cout<<"-------------------------------------------------------------------\n";
	string error_msg = "In order to run SPRINT-compute HSPs, type compute_HSPs and the following options: \n -p <protein_file> (required)\n -h <hsp_output_file_name> (required)\n -Thit <an integer, the threshold Thit > (optional, default 15) \n -Tsim <an integer, the threshold Tsim > (optional, default 35) \n -M <an integer, Scoring matrix. 1: PAM120, 2: BLOSUM80, 3: BLOSUM62> (optional, default PAM120)\n -add <new_protein_file previous_HSP_file_name> (optional, if new protein sequences are added and only HSPs in those sequences will be computed. New HSPs will be appended to previous_HSP_file. hsp_output_file_name will be ignored)\n" ;
	cout<<error_msg;
	cout<<"-------------------------------------------------------------------\n";
	for(int a = 0; a < argc; a ++){
		if(!strcmp(argv[a], "-p")){
			PROTEIN_FN = argv[a+1];
		}
		if(!strcmp(argv[a], "-h")){
			HSP_FN = argv[a+1];
		}
		if(!strcmp(argv[a], "-Thit")){
			Thit = atoi(argv[a+1]);
		}
		if(!strcmp(argv[a], "-Tsim")){
			T_kmer = atoi(argv[a+1]);
		}
		if(!strcmp(argv[a], "-M")){
			matrix_id = atoi(argv[a+1]);
		}
		if(!strcmp(argv[a], "-add")){
			ONLY_COMPUTE_NEW_PROTEIN = 1;
			NEW_PROTEIN_FN = argv[a+1];
			ORIGINAL_HSP_FN = argv[a+2];
		}
	}
	cout<<"PROTEIN_FN: "<<PROTEIN_FN<<endl;
	cout<<"HSP_FN: "<<HSP_FN<<endl;
	cout<<"Thit: "<<Thit<<endl;
	cout<<"Tsim: "<<T_kmer<<endl;
	if(matrix_id == 1){
		cout<<"Scoring matrix: PAM120\n";
		assign_matrix(BLOSUM80, PAM120);
	}
	else if(matrix_id == 2){
		cout<<"Scoring matrix: BLOSUM80\n";
		assign_matrix(BLOSUM80, BLOSUM80_1);
	}
	else if(matrix_id == 3){
		cout<<"Scoring matrix: BLOSUM62\n";
		assign_matrix(BLOSUM80, BLOSUM62);
	}

	calculate_B80_order();

	cout<<"-----------compute_hsps starts--------"<<endl;
	load_protein(PROTEIN_FN);
	cout<<"Number of Proteins: "<<num_protein<<endl;
	load_BLOSUM_convert(BLOSUM_convert);

	cout<<"-------initialize the hash tables-----"<<endl;
	HASH_TABLE ht0 = HASH_TABLE();
	HASH_TABLE ht1 = HASH_TABLE();
	HASH_TABLE ht2 = HASH_TABLE();
	HASH_TABLE ht3 = HASH_TABLE();
	#ifdef PARAL
	#pragma omp parallel 
	#endif
	{	
#ifdef PARAL	
#pragma omp sections
#endif
		{
#ifdef PARAL
#pragma omp section
#endif			
			{	
				ht0.creat_hash_table(seed_orig[0]);
			}
#ifdef PARAL
#pragma omp section
#endif			
			{
				ht1.creat_hash_table(seed_orig[1]);
			}
#ifdef PARAL			
#pragma omp section
#endif			
			{
				ht2.creat_hash_table(seed_orig[2]);
			}
#ifdef PARAL			
#pragma omp section
#endif			
			{
				ht3.creat_hash_table(seed_orig[3]);
			}
		}
	}

	cout<<"-----------Investigating the HSP table--------"<<endl;
	//build the HSP table
	PtoHSP hsp = PtoHSP();

	//load identical sub-sequences and store them into HSP table
	// cout<<"-----------load identical sub-sequences--------"<<endl;
	 
	// hsp.load_identi_sub_seq(SUB_SEQ_FN);
	// t = clock() - t;
	// printf ("load and compute identical sub-sequences took me (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);

	clock_t t = clock();
	cout<<"-----------Investigating the first hashtalbe (total: 4)"<<endl;
	hsp.load_hash_table(ht0);
	t = clock() - t;
	printf ("HSP table 0 took me (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);			
	
	t = clock();
	cout<<"-----------Investigating the second hashtalbe (total: 4)"<<endl;
	hsp.load_hash_table(ht1);
	t = clock() - t;
	printf ("HSP table 1 took me (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);			
	
	t = clock();
	cout<<"-----------Investigating the third hashtalbe (total: 4)"<<endl;
	hsp.load_hash_table(ht2);
	t = clock() - t;
	printf ("HSP table 2 took me (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);			
	
	t = clock();
	cout<<"-----------Investigating the fourth hashtalbe (total: 4)"<<endl;
	hsp.load_hash_table(ht3);
	t = clock() - t;
	printf ("HSP table 3 took me (%f seconds).\n",((float)t)/CLOCKS_PER_SEC);			

	hsp.print_HSP();
	cout<<"-----------compute_hsps finished--------"<<endl;

	return 0;
}
