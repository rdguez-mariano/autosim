// Compile with cmake (CMakeLists.txt is provided) or with the following lines in bash:
// g++ -c -fPIC libautosim.cpp -o libautosim.o
// g++ -shared -Wl,-soname,libautosim.so -o libautosim.so libautosim.o


#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <list>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <float.h>
#include <limits>
#include <cmath>


#define BIG_NUMBER_L1 2800.0f
#define BIG_NUMBER_L2 1000000000000.0f

class KPDescList
{
public:
    float *x, *y, *size, *angle, *scale, *desc;
    int len, desc_dim, *octavecode, *octave, *layer;
    static float distance_sift(const KPDescList &k1, const int idx1, const KPDescList &k2, const int idx2, float tdist, bool L2norm)
    {
      float dif;
      float distsq = 0.f;
      int len = k1.desc_dim;

      for (int i = 0; (i < len)&&(distsq <= tdist); i++)
      {
          dif = k1.desc[idx1*len + i] - k2.desc[idx2*len + i];
          if (L2norm)
              distsq += dif * dif;
          else
              distsq += std::abs(dif);
      }
      return distsq;
    }
    static double EuclideanDistance(const KPDescList &k1, int idx1, const KPDescList &k2, int idx2)
    {return(sqrt( pow(k1.x[idx1]-k2.x[idx2],2) + pow(k1.y[idx1]-k2.y[idx2],2) )); };
};

struct KPdescription {
  struct {float x,y;} pt;
  double	scale, size,
  angle, theta, t;
  int octave, layer;
  int idInList;
  KPdescription():theta(0.0),t(1.0),idInList(-1){};
static constexpr double labmda_descr = 6.0;

double DescRadius(){return(this->DescRadius(labmda_descr,false));}

double DescRadius(double factor, bool InPyr)
{
    if (InPyr)
        return( this->size*this->scale*factor*0.5 );
    else
        return( this->size*factor*0.5 );
}
static bool IntersectedDescriptors(KPdescription k1, KPdescription k2)
{
  double dist = sqrt( pow(k1.pt.x-k2.pt.x,2) + pow(k1.pt.y-k2.pt.y,2) );
  return(dist < 1.4142*( k1.DescRadius() + k2.DescRadius() ) );
}
};

// first = lhs; second = rhs
typedef std::pair<KPdescription,KPdescription> matching;
typedef std::vector<matching> matchingslist;



struct MatchClass;
struct GroupNode;
bool CheckAndLockGroup(GroupNode *g, bool doWait);
void UnLockGroup(GroupNode *g);

struct KPclass
{
  double x,y;
  double Sum_x, Sum_y;
  int N;
  std::vector< MatchClass* > ReferedMatches;

  static double EuclideanDistance(KPclass *k1,KPclass *k2) {return(sqrt( pow(k1->x-k2->x,2) + pow(k1->y-k2->y,2) )); };
  static double EuclideanDistance(KPclass *k1,KPdescription k2) {return(sqrt( pow(k1->x-k2.pt.x,2) + pow(k1->y-k2.pt.y,2) )); };

  void Update(KPclass* kp)
  {
    this->Sum_x += kp->Sum_x;
    this->Sum_y += kp->Sum_y;
    this->N += kp->N;
    x = Sum_x / N;
    y = Sum_y / N;
  };

  void RandomPerturbate(double radius)
  {
    this->Sum_x += myrand(radius)*N;
    this->Sum_y += myrand(radius)*N;
    x = Sum_x / N;
    y = Sum_y / N;
  };
  bool operator <(const KPclass & kp) {return ( (this->x<kp.x) || ((this->x==kp.x)&&(this->y<kp.y)) );};

  static double rho;
  bool operator ==(const KPclass & kp) {return (std::sqrt(std::pow(this->x-kp.x,2) + std::pow(this->y-kp.y,2))<KPclass::rho);};
  bool operator ==(const KPdescription & kp) {return (std::sqrt(std::pow(this->x-kp.pt.x,2) + std::pow(this->y-kp.pt.y,2))<KPclass::rho);};
  KPclass(double x0, double y0):x(x0),y(y0) {Sum_x=x0; Sum_y=y0; N=1;};
  KPclass(KPclass* kp){x = (kp->x);y = (kp->y);Sum_x = (kp->Sum_x);Sum_y = (kp->Sum_y); N = (kp->N);};

private:
  double myrand(double radius){return( radius*((double) rand() / (RAND_MAX)-0.5));};
  KPclass();
};
std::ostream& operator<<(std::ostream& os, const KPclass & kp){return (os<<"("<< kp.x << "," << kp.y <<")") ;};
double KPclass::rho = 4.0;


struct MatchClass
{
  KPclass * lhs, * rhs;
  double similarity; // similarity (0<s<1) between descriptors (not keypoints). lhs==rhs <--> similarity==1
  int Id;
  matching DataMatch;
  GroupNode *membership;
  bool operator <(const MatchClass & m) {return (this->similarity>m.similarity);};
  bool operator <=(const MatchClass & m) {return (this->similarity>=m.similarity);};
  bool operator ==(const MatchClass & m) {return ( ((*this->lhs==*m.lhs)&&(*this->rhs==*m.rhs)) || ((*this->lhs==*m.rhs)&&(*this->rhs==*m.lhs)) );};
  void RandomPerturbate(double radius){ this->lhs->RandomPerturbate(radius); this->rhs->RandomPerturbate(radius); };

  static bool PointerOrder(MatchClass* kp1,MatchClass * kp2) { return (*kp1 < *kp2); };

  MatchClass(KPclass *kp1, KPclass *kp2, double sim, matching data):similarity(sim),DataMatch(data) { this->lhs = kp1; this->rhs = kp2;};
  MatchClass(MatchClass *m){lhs=m->lhs;rhs=m->rhs;membership =m->membership;similarity=m->similarity;DataMatch=m->DataMatch;};

  bool CheckAndLock(bool doWait)
  {
    bool passed = false;
#pragma omp critical(lockregion)    
      passed = CheckAndLockGroup(membership,doWait); 
    return passed;  
  };

  void Unlock()
  {
  #pragma omp critical(unlockregion)
    {      
        UnLockGroup(membership);
    }
  }
  
private:
  MatchClass();  

};
std::ostream& operator<<(std::ostream& os, const MatchClass & m){return (os<<*m.lhs<<"<-"<<(round( m.similarity* 10.0) / 10)<<"->"<<*m.rhs);};


struct ElementNode
{
  MatchClass *Match;

  bool operator <(const ElementNode & g) {return (*this->Match<*g.Match);};
  bool operator ==(const ElementNode & g) {return (this->Match==g.Match);}; //Pointer equality
  bool operator ==(const MatchClass* m) {return (this->Match==m);}; //Pointer equality

  ElementNode(MatchClass *M){ this->Match = M; };
private:
  ElementNode();
};
std::ostream& operator<<(std::ostream& os, const ElementNode & e){return (os<<*e.Match);};


struct GroupNode
{
  std::list<GroupNode>::iterator thisGroupOnList; //pointer on list
  std::list<ElementNode> Interior;
  std::list<ElementNode> Exterior;
  std::vector<KPclass*> KPvec;

  int locked_by = 0;
  omp_nest_lock_t glock;

  GroupNode(){omp_init_nest_lock(& glock);};
  bool operator <(const GroupNode & g) {return (this->KPvec.size()<g.KPvec.size());};
  bool operator ==(const GroupNode & g) {return (this==&g);}; //Pointer equality

  //Use before merging on the GroupNode to destroy
  void InteriorUpdateMembership(GroupNode* g){for (std::list<ElementNode>::iterator it = this->Interior.begin();   it != this->Interior.end();   ++it) it->Match->membership = g;};
};
std::ostream& operator<<(std::ostream& os, const GroupNode & g)
{
  os<<"------> Interior  ("<<g.KPvec.size()<<" KPs)"<<std::endl;
  for(std::list<ElementNode>::const_iterator it = g.Interior.begin(); it != g.Interior.end(); ++it) {os<<*it<<" , ";};
  os<<"EoI"<<std::endl<<"------> Exterior"<<std::endl;
  for(std::list<ElementNode>::const_iterator it = g.Exterior.begin(); it != g.Exterior.end(); ++it) {os<<*it<<" , ";};return os<<"EoE";
};

bool CheckAndLockGroup(GroupNode *g, bool doWait)
{
  bool passed = false;
    if ( (g->locked_by==0) || doWait)
    {      
      omp_set_nest_lock(&(g->glock));
      g->locked_by++;                
      passed = true;
    }
    return passed; 
};

void UnLockGroup(GroupNode *g)
{
  g->locked_by--; 
  omp_unset_nest_lock(&(g->glock));
};


class GroupingStrategy
{
public:
  std::list<GroupNode> GroupList;
  std::vector<MatchClass*> WorldOfMatches;
  uint maxNumMatches = 6000;
  KPDescList list, listac;
  double lambda;

   GroupingStrategy(){lambda = 0.3;};

  std::vector<double> DistStatsOnMatches(const std::list<ElementNode> & le)
  {
    std::vector<double> vec(3);
    if (le.begin()==le.end())
      return(vec);
    vec[0]=KPclass::EuclideanDistance(le.begin()->Match->lhs,le.begin()->Match->rhs);
    vec[1]=0; vec[2]=0;
    double t; int count = 0;
    for(std::list<ElementNode>::const_iterator it = le.begin(); it != le.end(); ++it)
    {
      t = KPclass::EuclideanDistance(it->Match->lhs,it->Match->rhs);
      if (vec[0]>t) vec[0] = t;
      vec[1] += t; count++;
      if (vec[2]<t) vec[2] = t;
    }
    vec[1] = vec[1]/count;
    return(vec);
  }

  std::vector<double> DistStatsOnKPs(const std::vector<KPclass *> & kpvec)
  {
    std::vector<double> vec(3);
    if (kpvec.size()==0)
      return(vec);
    vec[0]=std::numeric_limits<double>::infinity();
    vec[1]=0; vec[2]=0;
    double t; int count = 0;
    for(uint i=1;i<kpvec.size();i++)
      for(uint j=0;j<i;j++)
        {
          t = KPclass::EuclideanDistance(kpvec[i],kpvec[j]);
          if (vec[0]>t) vec[0] = t;
          vec[1] += t; count++;
          if (vec[2]<t) vec[2] = t;
        }
    vec[1] = vec[1]/count;
    return(vec);
  }

  bool IntersectedGroupDescriptors(const GroupNode* g1, const GroupNode* g2)
  {
    for(std::list<ElementNode>::const_iterator it1 = g1->Interior.begin(); it1 != g1->Interior.end(); ++it1)
      for(std::list<ElementNode>::const_iterator it2 = g2->Interior.begin(); it2 != g2->Interior.end(); ++it2)
      {
        if ( (it1->Match->lhs!=it2->Match->lhs) && KPdescription::IntersectedDescriptors(it1->Match->DataMatch.first,it2->Match->DataMatch.first) )
          return(true);
        if ( (it1->Match->rhs!=it2->Match->rhs) && KPdescription::IntersectedDescriptors(it1->Match->DataMatch.second,it2->Match->DataMatch.second) )
          return(true);
        if ( (it1->Match->lhs!=it2->Match->rhs) && KPdescription::IntersectedDescriptors(it1->Match->DataMatch.first,it2->Match->DataMatch.second) )
          return(true);
        if ( (it1->Match->rhs!=it2->Match->lhs) && KPdescription::IntersectedDescriptors(it1->Match->DataMatch.second,it2->Match->DataMatch.first) )
          return(true);
      }
    return(false);
  }

  double Gcost(const GroupNode* g)
  {
    double I = 0.0;
    for(std::list<ElementNode>::const_iterator it = g->Interior.begin(); it != g->Interior.end(); ++it)
      {I += it->Match->similarity; };
        
    return(  1.0/I );
    // return(  1.0/g->KPvec.size() );
  };

  
  template <typename ListOrVectorOfMatches>
  static int IsMatchInVector(MatchClass* m, ListOrVectorOfMatches mvec)
  {
    bool found = false, alreadyfound = false;        
    // for(uint i=0;i<mvec.size();i++)
    for(typename ListOrVectorOfMatches::iterator it = mvec.begin(); it != mvec.end(); ++it)
      if (*it==m)
      {
        if (found)
          alreadyfound = true;
        found = true;
      }
    if (alreadyfound)
      return(2);
    if (found)
      return(1);
    return(0);
  }

   bool SanityCheck()
  {
    int intcount = 0;
    for(std::list<GroupNode>::iterator git = GroupList.begin(); git != GroupList.end(); ++git)
    {
      intcount += git->Interior.size();
      for (std::list<ElementNode>::iterator it = git->Interior.begin();   it != git->Interior.end();   ++it)
      {        
        int res = IsMatchInVector(it->Match, WorldOfMatches);          
            
        if (res!=1)
          std::cout<<"Interior santity check error "<<res<<std::endl;
      }
    }

    int extcount = 0;
    for(std::list<GroupNode>::iterator git = GroupList.begin(); git != GroupList.end(); ++git)
    {
      extcount += git->Exterior.size();
      for (std::list<ElementNode>::iterator it = git->Exterior.begin();   it != git->Exterior.end();   ++it)
      {
        int res = IsMatchInVector(it->Match, WorldOfMatches);         
            
        if (res!=1)
          std::cout<<"Ext santity check error "<<res<<std::endl;
               
        
        MatchClass* m = it->Match;
        res = 0;
        if (m->lhs!=0)         
        for (uint l=0;l<m->lhs->ReferedMatches.size();l++)
        {
          if (m->lhs->ReferedMatches[l]->lhs!=0)
            res += IsMatchInVector(m, m->lhs->ReferedMatches[l]->lhs->ReferedMatches);
          if (m->lhs->ReferedMatches[l]->rhs!=0)
            res += IsMatchInVector(m, m->lhs->ReferedMatches[l]->rhs->ReferedMatches);
          if (res==0)
            std::cout<<"LHS santity check error "<<res<<std::endl;          
        }          
        res = 0;
        if (m->rhs!=0)         
        for (uint l=0;l<m->rhs->ReferedMatches.size();l++)
        {
          if (m->rhs->ReferedMatches[l]->lhs!=0)
            res += IsMatchInVector(m, m->rhs->ReferedMatches[l]->lhs->ReferedMatches);
          if (m->rhs->ReferedMatches[l]->rhs!=0)
            res += IsMatchInVector(m, m->rhs->ReferedMatches[l]->rhs->ReferedMatches);
          if (res==0)
            std::cout<<"RHS santity check error "<<res<<std::endl;          
        }          
      }      
    }
  std::cout<<"WoM="<<WorldOfMatches.size()<<", Interior="<<intcount<<", Exterior="<<extcount<<std::endl;
  }

  double ThisCost()
  {
    double groupscost = 0;
    for(std::list<GroupNode>::iterator git = GroupList.begin(); git != GroupList.end(); ++git)
      groupscost += Gcost(&(*git));
    return(groupscost);
  };


 bool Add_OneElement_Group(MatchClass * m0)
  {
    MatchClass *m = new MatchClass(m0);
    KPclass* kp=Find_KP( m->lhs );
    if (kp!=0)
    {
      kp->Update(m->lhs);
      m->lhs = kp;
    }
    else
      m->lhs = new KPclass(m->lhs);

    kp=Find_KP( m->rhs );
    if (kp!=0)
    {
      kp->Update(m->rhs);
      m->rhs = kp;
    }
    else
      m->rhs = new KPclass(m->rhs);

    if(*m->lhs==*m->rhs || KPdescription::IntersectedDescriptors(m->DataMatch.first,m->DataMatch.second) )
    {
      delete m;
      return(false);
    }


    MatchClass* already_in = Find_Match(m);
    if (already_in!=0)
    {
      if (already_in->similarity < m->similarity)
        {
          already_in->similarity = m->similarity;
          already_in->DataMatch = m->DataMatch;
          //if ( !(*already_in->lhs == *m->lhs) )
          if ( KPclass::EuclideanDistance(already_in->lhs, m->lhs) > KPclass::EuclideanDistance(already_in->lhs, m->rhs))
            {
              KPclass* temp = already_in->lhs;
              already_in->lhs = already_in->rhs;
              already_in->rhs = temp;
            }
          if ( KPclass::EuclideanDistance(already_in->lhs, already_in->DataMatch.first) > KPclass::EuclideanDistance(already_in->lhs, already_in->DataMatch.second))
            std::cerr << "Add_OneElement_Group Error: already*_in->lhs and already_in->DataMatch.first were expected to be near each other with respect to DataMatch.second !" << '\n';
        }
      delete m;
      return(false);
    }
    WorldOfMatches.push_back(m);
    return( true );
  };

  void Initialize()
  {
    // ordering WorldOfMatches
    sort(WorldOfMatches.begin(), WorldOfMatches.end(), MatchClass::PointerOrder);

    if (maxNumMatches<WorldOfMatches.size())
      WorldOfMatches.resize(maxNumMatches);

    for (uint i=0; i < WorldOfMatches.size(); ++i)
    {
      MatchClass *m = WorldOfMatches[i];
      m->Id = i;
      GroupNode g; g.Interior.push_back(m);  //it calls constructor of ElementNode with MatchClass*
      g.Interior.begin()->Match->lhs->ReferedMatches.push_back(g.Interior.begin()->Match);
      g.Interior.begin()->Match->rhs->ReferedMatches.push_back(g.Interior.begin()->Match);
      g.Interior.sort();
      g.KPvec.push_back(m->lhs);
      g.KPvec.push_back(m->rhs);

      GroupList.push_back(g);
      std::list<GroupNode>::iterator it = --GroupList.end();
      it->thisGroupOnList = it;
      it->InteriorUpdateMembership(&(*it)); //&(*it) is a GroupeNode*
    }

    // Reconstruct all Exteriors from one element groups
    for(std::list<GroupNode>::iterator git = GroupList.begin(); git != GroupList.end(); ++git)
    {
      std::list<ElementNode>::iterator it = git->Interior.begin();
      if (it != git->Interior.end())
      {
        for (uint i=0;i<it->Match->lhs->ReferedMatches.size();i++)
          if (it->Match->lhs->ReferedMatches[i]!=it->Match)
            git->Exterior.push_back(it->Match->lhs->ReferedMatches[i]);
        for (uint i=0;i<it->Match->rhs->ReferedMatches.size();i++)
          if (it->Match->rhs->ReferedMatches[i]!=it->Match)
            git->Exterior.push_back(it->Match->rhs->ReferedMatches[i]);
      }
      git->Exterior.sort();
    }    
    std::cout<<"Unique Matches: "<<WorldOfMatches.size()<<std::endl;
  };

  KPclass* Find_KP(KPclass* kp)
  {
    for (uint i=0; i < WorldOfMatches.size(); ++i)
      {
        if (*kp==*(WorldOfMatches[i]->lhs))
          return (WorldOfMatches[i]->lhs);
        if (*kp==*(WorldOfMatches[i]->rhs))
          return (WorldOfMatches[i]->rhs);
      }
    return(0);
  };

  MatchClass* Find_Match(MatchClass* m)
  {
    for (uint i=0; i < WorldOfMatches.size(); ++i)
      if (*m==*WorldOfMatches[i])
        return(WorldOfMatches[i]);
    return(0);
  };

  void PrintWorldSimilarities(){for(uint i=0;i<WorldOfMatches.size();i++) std::cout<<WorldOfMatches[i]->similarity<<", ";std::cout<<"EoD"<<std::endl;};

  void Analyse();
  void AnalysePortions(int NumberOfPortions);
  static void write_images_matches(float* ipixels,int w1, int h1,const GroupNode & g, float * rgb, float * rgb_rich);
  void PrintGroups(bool write_groups);
  GroupNode* MergedCopy(GroupNode* g1, GroupNode* g2, MatchClass* gm);

};

GroupNode* GroupingStrategy::MergedCopy(GroupNode* g1, GroupNode* g2, MatchClass* m)
{
  GroupNode* g = new GroupNode();
  g->Interior.insert(g->Interior.end(),g1->Interior.begin(),g1->Interior.end());
  g->Interior.insert(g->Interior.end(),g2->Interior.begin(),g2->Interior.end());
  if(m!=0)    
    g->Interior.insert(g->Interior.end(),m->membership->Interior.begin(),m->membership->Interior.end());
  g->Interior.sort();
  g->Interior.unique();

  for (std::list<ElementNode>::iterator it = g->Interior.begin();   it != g->Interior.end();   ++it)
      {
        bool lfound = false, rfound = false;
        for(uint i=0;i<g->KPvec.size();i++)
        {
          if (g->KPvec[i]==it->Match->lhs)
            lfound = true;
          if (g->KPvec[i]==it->Match->rhs)
            rfound = true;
        }
        if (!lfound)
          g->KPvec.push_back(it->Match->lhs);
        if (!rfound)
          g->KPvec.push_back(it->Match->rhs);
      }


  g->Exterior.insert(g->Exterior.end(),g1->Exterior.begin(),g1->Exterior.end());
  g->Exterior.insert(g->Exterior.end(),g2->Exterior.begin(),g2->Exterior.end());

  if(m!=0)
    g->Exterior.insert(g->Exterior.end(),m->membership->Exterior.begin(),m->membership->Exterior.end()); 

  g->Exterior.sort();
  g->Exterior.unique();

 for(std::list<ElementNode>::iterator it = g->Interior.begin(); it != g->Interior.end(); ++it)
   g->Exterior.remove(*it);

  return(g);
}

void GroupingStrategy::Analyse()
{    
    double oldcost, newcost;
    for (int i =0; i<10;i++)
    {
      // SanityCheck();
      oldcost = ThisCost();
      AnalysePortions(8);
      newcost = ThisCost();
      if (oldcost==newcost)
        break;
      if (oldcost+0.00001<newcost)
        std::cerr<<"Error in function Analyse(): oldcost < newcost ("<<oldcost<<" < "<<newcost<<")"<<std::endl;
    }
    // for (int i =0; i<10;i++)
    // {
    //   // SanityCheck();
    //   oldcost = ThisCost();
    //   AnalysePortions(1);
    //   newcost = ThisCost();
    //   if (oldcost==newcost)
    //     break;
    //   if (oldcost+0.00001<newcost)
    //     std::cerr<<"Error in function Analyse(): oldcost < newcost ("<<oldcost<<" < "<<newcost<<")"<<std::endl;
    // }
    GroupList.sort();
}

void GroupingStrategy::AnalysePortions(int NumberOfPortions)
{
  double groupscost = ThisCost();

  // std::cout<<"Starting cost: "<<groupscost<<std::endl;
  // for (int p=0;p<NumberOfPortions;p++)
  // {
   std::cout<<"Starting cost: "<<groupscost<<std::endl;     
  omp_set_dynamic(0);
  omp_set_num_threads(NumberOfPortions);
#pragma omp parallel  
  {
    if (omp_get_num_threads() != NumberOfPortions)
      abort();
  int p = omp_get_thread_num();
  std::vector<GroupNode*> membership_lhs, membership_rhs;
  int id_min = floor(p*WorldOfMatches.size()/NumberOfPortions);
  int id_max = floor((p+1)*WorldOfMatches.size()/NumberOfPortions)-1;

  // #pragma omp parallel for firstprivate(groupscost)// schedule(static,2)
    for(int w=id_min;w<=id_max;w++)
    {
      MatchClass* m = WorldOfMatches[w];
      if (! m->CheckAndLock(false))
        continue;
    
      MatchClass* mm = 0; 
      GroupNode* gm = 0;
      std::list<ElementNode>::iterator itm, it2m;
      for (std::list<ElementNode>::iterator it = m->membership->Exterior.begin();   it != m->membership->Exterior.end();   ++it)  
      // for (std::list<ElementNode>::iterator it2 = std::next(it);   it2 != m->membership->Exterior.end();   ++it2)  
        {
          if (! it->Match->CheckAndLock(false))
            continue;
          
          if (false) //restrict to zone id_min - id_max
          {
            bool regionflag = true;
            for (std::list<ElementNode>::iterator it2 = it->Match->membership->Interior.begin();   it2 != it->Match->membership->Interior.end();   ++it2)  
              if (it2->Match->Id<id_min || it2->Match->Id>id_max)
                regionflag = false;
            if (regionflag==false)
              {
                it->Match->Unlock();
                continue;
              }
          }

          // Non intersected descriptors between both groups
          if (IntersectedGroupDescriptors(m->membership, it->Match->membership)) // Non Intersected groups
            {
              it->Match->Unlock();
              continue;
            }            

          // GroupNode* ug = MergedCopy(it2->Match->membership, it->Match->membership, m);
          GroupNode* ug = MergedCopy(m->membership, it->Match->membership, 0);
          
          double tug = Gcost(ug);             
          double tsgs = Gcost(m->membership);
          
          // if (it2->Match->membership == it->Match->membership)
            tsgs += Gcost(it->Match->membership);
          // else
          //   tsgs += Gcost(it->Match->membership) + Gcost(it2->Match->membership);

          if (m->membership == it->Match->membership)
            std::cerr<<"interior and exterior should not overlap.";

          if (tug<tsgs)
          {
            if (gm!=0)
            {
              mm->Unlock();
              delete gm;
            }
            mm = it->Match;
            gm = ug;
            itm = it;
            // it2m = it2;
  #pragma omp atomic write
            groupscost = groupscost - tsgs + tug;
          }
          else
          {
            it->Match->Unlock();
            delete ug;
          }
        }

        if (gm!=0)
  #pragma omp critical
        { 
          GroupNode* t1 = itm->Match->membership,*t2 = m->membership;                   

          GroupList.push_back(*gm);
          std::list<GroupNode>::iterator it = --GroupList.end();
          it->thisGroupOnList = it;
          it->InteriorUpdateMembership(&(*it)); 
          
          UnLockGroup(t1);
          UnLockGroup(t2);
          GroupList.erase(t1->thisGroupOnList);             
          GroupList.erase(t2->thisGroupOnList);            
        }
        else        
          m->Unlock();
    }
  }
  
  std::cout<<"Final cost: " <<ThisCost()<<std::endl;
}

void GroupingStrategy::PrintGroups(bool write_groups)
{
  GroupList.sort();
  std::vector<int> Cardinalities, Gcount;
  uint NoM = 0;

  if (GroupList.begin()!=GroupList.end())
  {
    int Count = 0;
    uint ccard = GroupList.begin()->KPvec.size();
    for(std::list<GroupNode>::iterator it = GroupList.begin(); it != GroupList.end(); ++it)
    {
      if (ccard!=it->KPvec.size())
      {
        Gcount.push_back(Count);
        Cardinalities.push_back(ccard);
        Count = 1;
        ccard = it->KPvec.size();
      }
      else
      {
        Count++;
      }
      NoM += it->Interior.size();
      if (write_groups)
        std::cout<<std::endl<<" *********** Group ID "<<Count<<" ************ "<<std::endl<<*it<<std::endl;
    }
    if (Count!=0)
    {
      Gcount.push_back(Count);
      Cardinalities.push_back(ccard);
    }
  }

    std::cout<<std::endl<<"Number of Groups for fixed C cardinalities"<<std::endl;
    for (uint i=0;i<Gcount.size();i++)
    std::cout<<"   C = "<<Cardinalities[i]<<" --> Number of Groups = "<<Gcount[i]<<std::endl;

    std::cout<<"Cost: " <<ThisCost()<<std::endl;

    if (NoM!=WorldOfMatches.size())
      std::cerr<< "Wrong final number of matches after Analysing. "<<NoM<<" in interiors against "<<WorldOfMatches.size()<<" in worldofmatches."<<std::endl;
}



#include "library.h"

void GroupingStrategy::write_images_matches(float* ipixels,int w1, int h1,const GroupNode & g, float * rgb, float * rgb_rich)
{
    int sq;

    int wo =  w1;
    int ho = h1;

    std::vector<float *> opixels, opixels_rich;
    for(int c=0;c<3;c++)
    {
        opixels.push_back(new float[wo*ho]);
        opixels_rich.push_back(new float[wo*ho]);
    }

    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) h1; j++)
            for(int i = 0; i < (int) w1; i++)
            {
                opixels[c][j*wo+i] = ipixels[j*w1+i];
                opixels_rich[c][j*wo+i] = ipixels[j*w1+i];
            }

    sq = 4;
    //////////////////////////////////////////////////////////////////// Draw matches
    float* colorlines = new float[3], *colordesc = new float[3];
    colorlines[0] = 1.0f;colorlines[1] = 1.0f; colorlines[2] = 1.0f;
    colordesc[0] = 1.0f;colordesc[1] = 250.0f; colordesc[2] = 1.0f;
    float value;
    bool trueKP = false;
    for(std::list<ElementNode>::const_iterator it = g.Exterior.begin(); it != g.Exterior.end(); ++it)
        for(int c=0;c<3;c++)
        {
            /* DRAWING SQUARES */
            if (trueKP)
            {
              matching * m = &(it->Match->DataMatch);
              draw_line(opixels[c],  round(m->first.pt.x), round(m->first.pt.y),
                        round(m->second.pt.x), round(m->second.pt.y), colorlines[c], wo, ho);

              draw_square(opixels[c],  round(m->first.pt.x)-sq, round(m->first.pt.y)-sq, 2*sq, 2*sq, colorlines[c], wo, ho);
              draw_square(opixels[c],  round(m->second.pt.x)-sq, round(m->second.pt.y) -sq, 2*sq, 2*sq, colorlines[c], wo, ho);
            }
            else
            {
              // matching * m = &(it->Match->DataMatch);
              draw_line(opixels[c],  round(it->Match->lhs->x), round(it->Match->lhs->y),
                        round(it->Match->rhs->x), round(it->Match->rhs->y), colorlines[c], wo, ho);

              draw_square(opixels[c],  round(it->Match->lhs->x)-sq, round(it->Match->lhs->y)-sq, 2*sq, 2*sq, colorlines[c], wo, ho);
              draw_square(opixels[c],  round(it->Match->rhs->x)-sq, round(it->Match->rhs->y) -sq, 2*sq, 2*sq, colorlines[c], wo, ho);
            }
        }

    sq = 2;
    for(std::list<ElementNode>::const_iterator it = g.Interior.begin(); it != g.Interior.end(); ++it)
        for(int c=0;c<3;c++)
        {
          value =  (float)((rand() % 150 + 50));
          matching * m = &(it->Match->DataMatch);
          if (trueKP)
          {
            draw_line(opixels[c],  round(m->first.pt.x), round(m->first.pt.y),
                      round(m->second.pt.x), round(m->second.pt.y), value, wo, ho);

            draw_square(opixels[c],  round(m->first.pt.x)-sq, round(m->first.pt.y)-sq, 2*sq, 2*sq, value, wo, ho);
            draw_square(opixels[c],  round(m->second.pt.x)-sq, round(m->second.pt.y) -sq, 2*sq, 2*sq, value, wo, ho);
          }
          else
          {
            draw_line(opixels[c],  round(it->Match->lhs->x), round(it->Match->lhs->y),
                      round(it->Match->rhs->x), round(it->Match->rhs->y), value, wo, ho);

            draw_square(opixels[c],  round(it->Match->lhs->x)-sq, round(it->Match->lhs->y)-sq, 2*sq, 2*sq, value, wo, ho);
            draw_square(opixels[c],  round(it->Match->rhs->x)-sq, round(it->Match->rhs->y) -sq, 2*sq, 2*sq, value, wo, ho);

          }
            /* DRAWING RICH KEYPOINTS */
            //draw_line(opixelsIMAS_rich[c],  round(matchings[i].first.x), round(matchings[i].first.y), round(matchings[i].second.x), round(matchings[i].second.y) + h1 + band_w, colorlines[c], wo, ho);
            draw_circle_affine(opixels_rich[c],wo,ho, m->first.pt.x, m->first.pt.y, m->first.angle*M_PI/180, m->first.DescRadius(), m->first.t, 1.0f, m->first.theta*M_PI/180, colordesc[c]);
            draw_circle_affine(opixels_rich[c],wo,ho, m->second.pt.x, m->second.pt.y, m->second.angle*M_PI/180, m->second.DescRadius(), m->second.t, 1.0f, m->second.theta*M_PI/180, colordesc[c]);
        }


    // std::ostringstream base;
    // base<<"imgs/CARD"<<g.KPvec.size()<<"-ID"<<count;

    // rgb = new float[wo*ho*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
                rgb[j+i*ho+c*(wo*ho)] = opixels[c][j*wo+i];
    // write_png_f32((base.str()+".png").c_str(), rgb, wo, ho, 3);

    // rgb_rich = new float[wo*ho*3];
    for(int c=0;c<3;c++)
        for(int j = 0; j < (int) ho; j++)
            for(int i = 0; i < (int) wo; i++)
                rgb_rich[j+i*ho+c*(wo*ho)] = opixels_rich[c][j*wo+i];
    // write_png_f32((base.str()+"_rich.png").c_str(), rgb, wo, ho, 3);

    for(int c=0;c<3;c++)
    {
        delete[] opixels[c]; /*memcheck*/
        delete[] opixels_rich[c]; /*memcheck*/
    }
}


static inline void unpackOctave(int octavecode, int& octave, int& layer, float& scale)
{
  octave = octavecode & 255;
  layer = (octavecode >> 8) & 255;
  octave = octave < 128 ? octave : (-128 | octave);
  scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}

// for debuging pourposes
void write_input_data(float* x, float *y, int *octcode,float *size,float *angle, int len, int desc_dim, float* desc, bool am_i_ac)
{
  std::ofstream file_obj;
  if (am_i_ac)
    file_obj.open("save_ac.dat",std::ios::binary);        
  else
    file_obj.open("save.dat",std::ios::binary);
  file_obj.write(reinterpret_cast<char*>(& len),sizeof(int));
  file_obj.write(reinterpret_cast<char*>(& desc_dim),sizeof(int));
  file_obj.write(reinterpret_cast<char*>(x),len*sizeof(float));
  file_obj.write(reinterpret_cast<char*>(y),len*sizeof(float));
  file_obj.write(reinterpret_cast<char*>(octcode),len*sizeof(int));
  file_obj.write(reinterpret_cast<char*>(size),len*sizeof(float));
  file_obj.write(reinterpret_cast<char*>(angle),len*sizeof(float));
  file_obj.write(reinterpret_cast<char*>(desc),len*desc_dim*sizeof(float));      
  file_obj.close();
}

void load_input_data(float* &x, float *&y, int *&octcode,float *&size,float *&angle, int &len, int &desc_dim, float* &desc, bool am_i_ac)
{
  std::ifstream file_obj;
  if (am_i_ac)
    file_obj.open("save_ac.dat",std::ios::in | std::ios::binary);        
  else
    file_obj.open("save.dat",std::ios::in | std::ios::binary);
  file_obj.read(reinterpret_cast<char*>(& len),sizeof(int));
  file_obj.read(reinterpret_cast<char*>(& desc_dim),sizeof(int));  
  x = new float[len];
  y = new float[len];
  angle = new float[len];
  size = new float[len];
  desc = new float[len*desc_dim];
  octcode = new int[len];

  file_obj.read(reinterpret_cast<char*>(x),len*sizeof(float));    
  file_obj.read(reinterpret_cast<char*>(y),len*sizeof(float));
  file_obj.read(reinterpret_cast<char*>(octcode),len*sizeof(int));
  file_obj.read(reinterpret_cast<char*>(size),len*sizeof(float));
  file_obj.read(reinterpret_cast<char*>(angle),len*sizeof(float));
  file_obj.read(reinterpret_cast<char*>(desc),len*desc_dim*sizeof(float));      
  file_obj.close();
}



// Define C functions for the C++ class - as ctypes can only talk to C...
extern "C"
{
    GroupingStrategy* New_GS(float rho, int maxNumMatches)
    {
      GroupingStrategy* gs = new GroupingStrategy; 
      gs->maxNumMatches= maxNumMatches; KPclass::rho = (double) rho; 
      return(gs);
    }

    void Add_match(GroupingStrategy* gs, float sim, int id1, float x1, float y1,int o1,float s1,float a1, int id2, float x2, float y2,int o2,float s2,float a2)
    {
      KPdescription k1, k2;
      int octave, layer;
      float scale;

      unpackOctave(o1, octave, layer, scale);
      k1.pt.x = x1;
      k1.pt.y = y1;
      k1.size = (double) s1;
      k1.angle = (double) a1;
      k1.scale = (double) scale;
      k1.layer = (double) layer;
      k1.octave = (double) octave;
      KPclass lhs_kp(x1,y1);

      unpackOctave(o2, octave, layer, scale);
      k2.pt.x = x2;
      k2.pt.y = y2;
      k2.size = (double) s2;
      k2.angle = (double) a2;
      k2.scale = (double) scale;
      k2.layer = (double) layer;
      k2.octave = (double) octave;
      KPclass rhs_kp(x2,y2);

      MatchClass m(&lhs_kp,&rhs_kp, (double) sim, matching(k1,k2));
      gs->Add_OneElement_Group(&m);
    }



    void Initialize(GroupingStrategy* gs) {gs->Initialize();}
    void Analyse(GroupingStrategy* gs) {gs->Analyse();}
    void PrintGroups(GroupingStrategy* gs)
    {
      gs->PrintGroups(false);
    }

    GroupNode* LastGroup(GroupingStrategy* gs)
    {
      if (gs->GroupList.begin()!=gs->GroupList.end())
        return(&*(--gs->GroupList.end()));
      else
        return(0);
    }

    GroupNode* FirstGroup(GroupingStrategy* gs)
    {
      if (gs->GroupList.begin()!=gs->GroupList.end())
        return(&*gs->GroupList.begin());
      else
        return(0);
    }

    GroupNode* NextGroup(GroupingStrategy* gs, GroupNode* g)
    {
      if (g!=0 && ++g->thisGroupOnList!=gs->GroupList.end())
        return(&*(++g->thisGroupOnList));
      else
        return(0);
    }

    GroupNode* PrevGroup(GroupingStrategy* gs, GroupNode* g)
    {
      if (g!=0 && g->thisGroupOnList!=gs->GroupList.begin())
        return(&*(--g->thisGroupOnList));
      else
        return(0);
    }

    int NumberOfMatches(GroupNode* g, bool Interior)
    {
      if (g!=0)
      {
        if (Interior)
          return g->Interior.size();
        else
          return g->Exterior.size();
      }
      else
        return(0);
    }

    int NumberOfKPs(GroupNode* g)
    {
      if (g!=0)
      {
          return g->KPvec.size();
      }
      else
        return(0);
    }

    void GetMatches(GroupNode* g, float* arr, bool Interior, bool trueKP)
    {
      int data_len = 5;
      if (trueKP)
        data_len = 7;
      if (g!=0)
        {
          std::list<ElementNode>::const_iterator itstart, itend;
          if (Interior)
            {
              itstart = g->Interior.begin();
              itend = g->Interior.end();
            }
          else
            {
              itstart = g->Exterior.begin();
              itend = g->Exterior.end();
            }
          int fcc = 0;
          if (trueKP)          
            for(std::list<ElementNode>::const_iterator it = itstart; it != itend; ++it)
            {
              matching * m = &(it->Match->DataMatch);
              arr[fcc*data_len] = (float) it->Match->lhs->x;
              arr[fcc*data_len+1] = (float) it->Match->lhs->y;
              arr[fcc*data_len+2] = (float)it->Match->rhs->x;
              arr[fcc*data_len+3] = (float) it->Match->rhs->y;
              arr[fcc*data_len+4] = (float) it->Match->similarity;              
              arr[fcc*data_len+5] = (float) m->first.idInList;
              arr[fcc*data_len+6] = (float) m->second.idInList;
              fcc++;
            }
          else
            for(std::list<ElementNode>::const_iterator it = itstart; it != itend; ++it)
            {
              arr[fcc*data_len] = (float) it->Match->lhs->x;
              arr[fcc*data_len+1] = (float) it->Match->lhs->y;
              arr[fcc*data_len+2] = (float)it->Match->rhs->x;
              arr[fcc*data_len+3] = (float) it->Match->rhs->y;
              arr[fcc*data_len+4] = (float) it->Match->similarity;
              fcc++;
            }                      
        }
    }

    void getImagesFromGroup(GroupNode* g, float* img, int w, int h, float* rgb, float* rgb_rich)
    {
      GroupingStrategy::write_images_matches(img, w, h, *g, rgb, rgb_rich);
    }

    void bind2list(KPDescList* list, float *x, float *y, int *octcode,float *size,float *angle, int len, int desc_dim, float* desc)
    {
      list->x = x;
      list->y = y;
      list->size = size;
      list->angle = angle;
      list->len = len;
      list->desc_dim = desc_dim;
      list->desc = desc;

      int octave, layer;
      float scale;
      list->octave = new int[len];
      list->layer = new int[len];
      list->scale = new float[len];

      for (int i=0; i<len; i++)
      {
          unpackOctave(octcode[i], octave, layer, scale);
          list->octave[i] = octave;
          list->layer[i] = layer;
          list->scale[i] = scale;
      }
    }

    void Bind_KPs(GroupingStrategy* gs, float *x, float *y, int *octcode,float *size,float *angle, int len, int desc_dim, float* desc, bool am_i_ac)    {
      if (am_i_ac)
        bind2list(&gs->listac, x, y, octcode, size, angle, len, desc_dim, desc);
      else
        bind2list(&gs->list, x, y, octcode, size, angle, len, desc_dim, desc);
      // write_input_data(x, y, octcode, size, angle, len, desc_dim, desc,am_i_ac);
    }

    void ACMatcher(GroupingStrategy* gs, float matchratio)
    {
      double	dsq, distsq1;
      matchingslist matchings;
      std::vector<double> ac_distance;
      bool flagL2 = true;
      matchratio = matchratio*matchratio;
      int countmatches = 0;

      KPDescList listac = gs->listac;
      #pragma omp parallel for shared(matchings,ac_distance) firstprivate(listac) private(dsq,distsq1)
      for (int i=0; i< (int) gs->list.len; i++)
      {
        // minimal distance to the a-contrario descriptors
        // std::cout<<i<<std::endl;
        distsq1 = (double)KPDescList::distance_sift(gs->list,i, listac,0,BIG_NUMBER_L2,flagL2);
        for (int j=1; j< (int) listac.len; j++)
        {
          dsq = (double)KPDescList::distance_sift(gs->list,i, listac,j,distsq1,flagL2);
          if (dsq < distsq1)
            distsq1 = dsq;
        }
        // std::cout<<i<<std::endl;
        //see all others
        for (int j=0; j< (int) gs->list.len; j++)
        {
          if((i!=j)&&(KPDescList::EuclideanDistance(gs->list,i, gs->list,j)>KPclass::rho))
          {
            dsq = (double)KPDescList::distance_sift(gs->list,i, gs->list,j,distsq1,flagL2);
            if (dsq < matchratio*distsq1)
            {
              
                // std::cout<<i<<"<->"<<j<<std::endl;
                countmatches++;
                KPdescription k1, k2;

                k1.pt.x = gs->list.x[i];
                k1.pt.y = gs->list.y[i];
                k1.size = (double) gs->list.size[i];
                k1.angle = (double)gs->list.angle[i];
                k1.scale = (double) gs->list.scale[i];
                k1.layer = (int) gs->list.layer[i];
                k1.octave = (int) gs->list.octave[i];
                k1.idInList = i;

                KPclass lhs_kp(k1.pt.x,k1.pt.y);

                k2.pt.x = gs->list.x[j];
                k2.pt.y = gs->list.y[j];
                k2.size = (double) gs->list.size[j];
                k2.angle = (double)gs->list.angle[j];
                k2.scale = (double) gs->list.scale[j];
                k2.layer = (int) gs->list.layer[j];
                k2.octave = (int) gs->list.octave[j];
                k2.idInList = j;

                KPclass rhs_kp(k2.pt.x,k2.pt.y);

                double sim = ( dsq/distsq1>1 ? 0.0 : 1 - dsq/distsq1 );

                MatchClass m(&lhs_kp,&rhs_kp, sim,matching(k1,k2));
              #pragma omp critical
              {                
                gs->Add_OneElement_Group(&m);
              }
            }
          }
        }
      }
      std::cout<<"Found Matches: "<<countmatches<<std::endl;
    }


}

int main()
{
  GroupingStrategy* gs = new GroupingStrategy;
  
  float *x, *y, *size, *angle, *desc;
  int *octcode, len, desc_dim;
  
  bool am_i_ac = true;
  load_input_data(x, y, octcode, size, angle, len, desc_dim, desc,am_i_ac);
  bind2list(&gs->listac, x, y, octcode, size, angle, len, desc_dim, desc);
  am_i_ac = false;
  load_input_data(x, y, octcode, size, angle, len, desc_dim, desc,am_i_ac);
  bind2list(&gs->list, x, y, octcode, size, angle, len, desc_dim, desc);

  // int i = len - 1;
  // std::cout << "Element " << i << ": " << x[i] <<", " 
  // << y[i] <<", " << angle[i]<<", " <<  size[i] <<", "<<
  //  desc[i*desc_dim-51] << std::endl;
  
  ACMatcher(gs, 0.8f);
  gs->Initialize();
  gs->Analyse();
  gs->PrintGroups(false);
}