#include <fstream>
#include <iomanip>

void write_cp_times_header(std::ofstream& st, int compare){
  st << std::left << std::setw(15) << "Config";
  st << std::left << std::setw(15) << "Raw";
  st << std::left << std::setw(15) << "Decomp-exec";
  st << std::left << std::setw(15) << "Disc-exec";
  st << std::left << std::setw(15) << "Decomp-comp";
  st << std::left << std::setw(15) << "Disc-comp";
  st << std::left << std::setw(15) << "Decomp-compk";
  st << std::left << std::setw(15) << "Disc-compk";
  if (compare==1){
    st << std::left << std::setw(15) << "Decomp-synch";
    st << std::left << std::setw(15) << "Disc-synch";
  }
  st << std::left << std::setw(15) << "Decomp-comm";
  st << std::left << std::setw(15) << "Disc-comm";
  st << std::endl;
}
void write_cp_costs_header(std::ofstream& st, int compare){
  st << std::left << std::setw(15) << "Config";
  st << std::left << std::setw(15) << "Decomp-BSPcomm";
  st << std::left << std::setw(15) << "Decomp-ABcomm";
  st << std::left << std::setw(15) << "Decomp-BSPsynch";
  st << std::left << std::setw(15) << "Decomp-ABsynch";
  st << std::left << std::setw(15) << "Decomp-comp";
  st << std::endl;
}
void write_cross_times_header(std::ofstream& st, int compare){
  st << std::left << std::setw(15) << "Config";
  st << std::left << std::setw(15) << "cp-exec";
  st << std::left << std::setw(15) << "pp-exec";
  st << std::left << std::setw(15) << "vol-exec";
  st << std::left << std::setw(15) << "cp-comp";
  st << std::left << std::setw(15) << "pp-comp";
  st << std::left << std::setw(15) << "vol-comp";
  st << std::left << std::setw(15) << "cp-compk";
  st << std::left << std::setw(15) << "pp-compk";
  st << std::left << std::setw(15) << "vol-compk";
  st << std::left << std::setw(15) << "cp-comm";
  st << std::left << std::setw(15) << "pp-comm";
  st << std::left << std::setw(15) << "vol-comm";
  st << std::left << std::setw(15) << "cp-synch";
  st << std::left << std::setw(15) << "pp-synch";
  st << std::left << std::setw(15) << "vol-synch";
  st << std::endl;
}
void write_cross_costs_header(std::ofstream& st, int compare){
  st << std::left << std::setw(15) << "Config";
  st << std::left << std::setw(15) << "cp-comm";
  st << std::left << std::setw(15) << "pp-comm";
  st << std::left << std::setw(15) << "vol-comm";
  st << std::left << std::setw(15) << "cp-synch";
  st << std::left << std::setw(15) << "pp-synch";
  st << std::left << std::setw(15) << "vol-synch";
  st << std::left << std::setw(15) << "cp-comp";
  st << std::left << std::setw(15) << "pp-comp";
  st << std::left << std::setw(15) << "vol-comp";
  st << std::endl;
}

void write_cross_info(std::ofstream& st1, std::ofstream& st2, int compare,
                      int configuration_id, std::vector<float>& cp,
                      std::vector<float>& pp, std::vector<float>& vol){
  st1 << std::left << std::setw(15) << configuration_id;
  st1 << std::left << std::setw(15) << cp[cp.size()-1];
  st1 << std::left << std::setw(15) << pp[pp.size()-1];
  st1 << std::left << std::setw(15) << vol[vol.size()-1];
  st1 << std::left << std::setw(15) << cp[cp.size()-3];
  st1 << std::left << std::setw(15) << pp[pp.size()-3];
  st1 << std::left << std::setw(15) << vol[vol.size()-3];
  st1 << std::left << std::setw(15) << cp[cp.size()-2];
  st1 << std::left << std::setw(15) << pp[pp.size()-2];
  st1 << std::left << std::setw(15) << vol[vol.size()-2];
  st1 << std::left << std::setw(15) << cp[cp.size()-5];
  st1 << std::left << std::setw(15) << pp[pp.size()-5];
  st1 << std::left << std::setw(15) << vol[vol.size()-5];
  st1 << std::left << std::setw(15) << cp[cp.size()-4];
  st1 << std::left << std::setw(15) << pp[pp.size()-4];
  st1 << std::left << std::setw(15) << vol[vol.size()-4];
  st1 << std::endl;

  st2 << std::left << std::setw(15) << configuration_id;
  st2 << std::left << std::setw(15) << cp[0];
  st2 << std::left << std::setw(15) << pp[0];
  st2 << std::left << std::setw(15) << vol[0];
  st2 << std::left << std::setw(15) << cp[1];
  st2 << std::left << std::setw(15) << pp[1];
  st2 << std::left << std::setw(15) << vol[1];
  st2 << std::left << std::setw(15) << cp[2];
  st2 << std::left << std::setw(15) << pp[2];
  st2 << std::left << std::setw(15) << vol[2];
  st2 << std::endl;
}
void write_cp_info(std::ofstream& st1, std::ofstream& st2, int compare,
                   int configuration_id, float exec_time,
                   std::vector<float>& decomp_cp_info, std::vector<float>& disc_cp_info){
  st1 << std::left << std::setw(15) << configuration_id;
  st1 << std::left << std::setw(15) << exec_time;
  st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-1];
  st1 << std::left << std::setw(15) << disc_cp_info[disc_cp_info.size()-1];
  st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-3];
  st1 << std::left << std::setw(15) << disc_cp_info[disc_cp_info.size()-3];
  st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-2];
  st1 << std::left << std::setw(15) << disc_cp_info[disc_cp_info.size()-2];
  if (compare==1){
    st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-4];
    st1 << std::left << std::setw(15) << -1;// Synchronization not tracked
    st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-5];
    st1 << std::left << std::setw(15) << disc_cp_info[disc_cp_info.size()-4];
  } else{
    st1 << std::left << std::setw(15) << decomp_cp_info[decomp_cp_info.size()-4];
    st1 << std::left << std::setw(15) << disc_cp_info[disc_cp_info.size()-4];
  }
  st1 << std::endl;

  if (compare==1){
    st2 << std::left << std::setw(15) << configuration_id;
    st2 << std::left << std::setw(15) << decomp_cp_info[0];
    st2 << std::left << std::setw(15) << decomp_cp_info[1];
    st2 << std::left << std::setw(15) << decomp_cp_info[2];
    st2 << std::left << std::setw(15) << decomp_cp_info[3];
    st2 << std::left << std::setw(15) << decomp_cp_info[4];
    st2 << std::endl;
  }
}
