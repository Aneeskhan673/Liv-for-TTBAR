/***************************************************************
 * Combined SME Analysis for Lorentz Invariance Violation (LIV)  *
 * in Top Quark Production                                      *
 *                                                             *
 * This code analyzes potential LIV effects in tt̄ production    *
 * by comparing Standard Model predictions with SME-modified     *
 * matrix elements. Includes sidereal time dependence analysis. *
 ***************************************************************/

//-----------------------//
//  1. HEADER INCLUDES   //
//-----------------------//
#include <iostream>
#include <cmath>
#include "TFile.h"
#include "TTree.h"
#include "TLorentzVector.h"
#include "TMatrixD.h"
#include "TRandom3.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"
#include "classes/DelphesClasses.h"

//-----------------------//
//  2. PHYSICS CONSTANTS //
//-----------------------//
const double mt = 172.76;      // Top quark mass (GeV)
const double gs = 1.217;       // Strong coupling constant (α_s)
const double mW = 80.379;      // W boson mass (GeV)
const double Gamma_t = 1.41;   // Top quark width (GeV)
const double Gamma_W = 2.085;  // W boson width (GeV)
const double chi = 43.8 * M_PI / 180.0;  // Colatitude of CERN (radians)

//-----------------------//
//  3. FUNCTION DECLARATIONS //
//-----------------------//
void calculateMandelstam(const TLorentzVector&, const TLorentzVector&, 
                         const TLorentzVector&, const TLorentzVector&, 
                         double&, double&, double&);
double P2gSM_Full(double, double, double, double, double);
double DeltaProduction_Full(double, double, double, double, double, 
                               const TMatrixD&, const TLorentzVector&, 
                               const TLorentzVector&, const TLorentzVector&,
                               const TLorentzVector&);
double computeLIVWeight(const TLorentzVector&, const TLorentzVector&,
                        const TLorentzVector&, const TLorentzVector&,
                        const TMatrixD&);
void GeneratePlots(TTree*);
TMatrixD computeRotationMatrix(double, double);
TMatrixD rotateCmuNu(const TMatrixD&, const TMatrixD&);
double Contract(const TMatrixD&, const TLorentzVector&, const TLorentzVector&);
void PrintFourVector(const TLorentzVector&, const char* name = "Vector");

//-----------------------//
//  4. PHYSICS CALCULATIONS //  
//-----------------------//

// Helper function: Contract matrix c_{mu nu} with two vectors: c^{mu nu} * a_mu * b_nu
double Contract(const TMatrixD& cMatrix, const TLorentzVector& a, const TLorentzVector& b) {
    double result = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            result += cMatrix(mu, nu) * a[mu] * b[nu];
        }
    }
    return result;
}

// Debugging helper
void PrintFourVector(const TLorentzVector& v, const char* name) {
    std::cout << name << ": (E=" << v.E() << ", px=" << v.Px() << ", py=" << v.Py() << ", pz=" << v.Pz() << ")" << std::endl;
}

// Calculate Mandelstam variables s, t, u
void calculateMandelstam(const TLorentzVector& p1, const TLorentzVector& p2,
                         const TLorentzVector& p3, const TLorentzVector& p4,
                         double& s, double& t, double& u) {
    s = (p1 + p2).M2();  // s = (p1 + p2)^2
    t = (p1 - p3).M2();  // t = (p1 - p3)^2 
    u = (p1 - p4).M2();  // u = (p1 - p4)^2
}

// Individual SM terms for gg->ttbar (Eq. 15)
double M_ss_SM(double gs, double s, double t, double u, double mt) {
    return (3.0 * gs*gs*gs*gs / 4.0) * ((t - mt*mt) * (u - mt*mt)) / (s*s);
}
double M_tt_SM(double gs, double s, double t, double u, double mt) {
    double numerator = (t - mt*mt)*(u - mt*mt) - 2*mt*mt*(t + mt*mt);
    return (gs*gs*gs*gs / 6.0) * (numerator) / ((t - mt*mt)*(t - mt*mt));
}
double M_uu_SM(double gs, double s, double t, double u, double mt) {
    double numerator = (u - mt*mt)*(t - mt*mt) - 2*mt*mt*(u + mt*mt);
    return (gs*gs*gs*gs / 6.0) * (numerator) / ((u - mt*mt)*(u - mt*mt));
}
double M_st_SM(double gs, double s, double t, double u, double mt) {
    double numerator = (t - mt*mt)*(u - mt*mt) + mt*mt*(u - t);
    return (3.0 * gs*gs*gs*gs / 8.0) * (numerator) / (s * (t - mt*mt));
}
double M_su_SM(double gs, double s, double t, double u, double mt) {
    double numerator = (u - mt*mt)*(t - mt*mt) + mt*mt*(t - u);
    return (3.0 * gs*gs*gs*gs / 8.0) * (numerator) / (s * (u - mt*mt));
}
double M_tu_SM(double gs, double s, double t, double u, double mt) {
    double numerator = - mt*mt * (s - 4*mt*mt);
    return (gs*gs*gs*gs / 24.0) * (numerator) / ((t - mt*mt) * (u - mt*mt));
}

// Full SM matrix element squared for gg->ttbar (Sum of Eq. 15)
double P2gSM_Full(double gs, double s, double t, double u, double mt) {
    return M_ss_SM(gs, s, t, u, mt) +
           M_tt_SM(gs, s, t, u, mt) +
           M_uu_SM(gs, s, t, u, mt) +
           M_st_SM(gs, s, t, u, mt) +
           M_su_SM(gs, s, t, u, mt) +
           M_tu_SM(gs, s, t, u, mt);
}

// LIV Corrections: Propagator Insertions (Eq. 25-30)
double delta_M_ss_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (4.0 * s*s);
    double contraction = s * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t))
                       + (t - mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p2, p_t))
                       + (u - mt*mt) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p2, p_tbar));
    return preFactor * contraction;
}
double delta_M_tt_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (6.0 * pow(t - mt*mt, 3));
    double term1 = (-t*t + t*u - mt*mt*u + 3*mt*mt*t - 10*pow(mt,4)) * 2 * Contract(cMatrix, p1, p2);
    double term2 = (t*t + t*u - mt*mt*u - 9*mt*mt*t) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    double term3 = (-t*t - t*u + mt*mt*u + 5*mt*mt*t + 4*pow(mt,4)) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2));
    return preFactor * (term1 + term2 + term3);
}
double delta_M_uu_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (6.0 * pow(u - mt*mt, 3));
    double term1 = (-u*u + u*t - mt*mt*t + 3*mt*mt*u - 10*pow(mt,4)) * 2 * Contract(cMatrix, p1, p2);
    double term2 = (u*u + u*t - mt*mt*t - 9*mt*mt*u) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    double term3 = (-u*u - u*t + mt*mt*t + 5*mt*mt*u + 4*pow(mt,4)) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2));
    return preFactor * (term1 + term2 + term3);
}
double delta_M_st_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (32.0 * s * pow(t - mt*mt, 2));
    double term1 = (2*t*s - (t + mt*mt)*(u - mt*mt)) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2));
    double term2a = (t - mt*mt) * ( (3*t - 5*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2)) );
    double term2b = (t - mt*mt) * ( (t + 3*u - 8*mt*mt) * (Contract(cMatrix, p1, p2) - Contract(cMatrix, p2, p1) - Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) - Contract(cMatrix, p_t, p2)) );
    double term3 = -2 * (8*pow(mt,4) + (t - 3*mt*mt)*(3*t + u)) * 2 * Contract(cMatrix, p1, p2);
    double term4 = 4 * (2*t*u - 3*mt*mt*t - mt*mt*u + 2*pow(mt,4)) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    return preFactor * (term1 + term2a + term2b + term3 + term4);
}
double delta_M_su_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (32.0 * s * pow(u - mt*mt, 2));
    double term1 = (2*u*s - (u + mt*mt)*(t - mt*mt)) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2));
    double term2a = (u - mt*mt) * ( (3*u - 5*mt*mt) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2)) );
    double term2b = (u - mt*mt) * ( (u + 3*t - 8*mt*mt) * (Contract(cMatrix, p1, p2) - Contract(cMatrix, p2, p1) - Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) - Contract(cMatrix, p_tbar, p2)) );
    double term3 = -2 * (8*pow(mt,4) + (u - 3*mt*mt)*(3*u + t)) * 2 * Contract(cMatrix, p1, p2);
    double term4 = 4 * (2*t*u - 3*mt*mt*u - mt*mt*t + 2*pow(mt,4)) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    return preFactor * (term1 + term2a + term2b + term3 + term4);
}
double delta_M_tu_Prop(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (24.0 * pow(u - mt*mt, 2) * pow(t - mt*mt, 2));
    double term1 = (2*s + mt*mt) * (t - mt*mt) * (u - mt*mt) * (2*Contract(cMatrix, p1, p2) - Contract(cMatrix, p_t, p_tbar) - Contract(cMatrix, p_tbar, p_t));
    double term2a = mt*mt * (s*s - 7*mt*mt*s - 3*t*u + 3*pow(mt,4)) * (2*Contract(cMatrix, p1, p2) + Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    double term2b = - mt*mt * (t - mt*mt) * (t - u + 4*mt*mt) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2));
    double term2c = mt*mt * (u - mt*mt) * (t - u - 4*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2));
    return preFactor * (term1 + term2a + term2b + term2c);
}

// LIV Corrections: Vertex Insertions (Eq. 31-36)
double delta_M_ss_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (4.0 * s*s);
    TLorentzVector p1_minus_p2 = p1 - p2;
    double contraction =
    t * ( Contract(cMatrix,p_t,p1) + Contract(cMatrix,p_tbar,p2)
        - (Contract(cMatrix,p1,p2) + Contract(cMatrix,p2,p1)) )
  + u * ( Contract(cMatrix,p_t,p2) + Contract(cMatrix,p_tbar,p1)
        - (Contract(cMatrix,p1,p2) + Contract(cMatrix,p2,p1)) )
  - mt*mt * Contract(cMatrix, p1_minus_p2, p1_minus_p2);
    return preFactor * contraction;
}
double delta_M_tt_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (3.0 * pow(t - mt*mt, 2));
    double term1 = (t - 3*mt*mt) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2));
    double term2 = 4*mt*mt * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    return preFactor * (term1 + term2);
}
double delta_M_uu_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (3.0 * pow(u - mt*mt, 2));
    double term1 = (u - 3*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2));
    double term2 = 4*mt*mt * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    return preFactor * (term1 + term2);
}
double delta_M_st_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (32.0 * s * (t - mt*mt));
    double term1 = 2*(s + 4*mt*mt) * 2 * Contract(cMatrix, p1, p2);
    double term2 = (4*t + 3*u - 13*mt*mt) * (Contract(cMatrix, p1, p_t) + Contract(cMatrix, p_t, p1) + Contract(cMatrix, p2, p_tbar) + Contract(cMatrix, p_tbar, p2));
    double term3 = 4*(t - u) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    double term4 = (-2*t - 3*u + 7*mt*mt) * (Contract(cMatrix, p_t, p1) + Contract(cMatrix, p_t, p2));
    double term5 = (3*u - 9*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p2, p_tbar));
    return preFactor * (term1 + term2 + term3 + term4 + term5);
}
double delta_M_su_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (3.0 * gs*gs*gs*gs) / (32.0 * s * (u - mt*mt));
    double term1 = 2*(s + 4*mt*mt) * 2 * Contract(cMatrix, p1, p2);
    double term2 = (4*u + 3*t - 13*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p_tbar, p1) + Contract(cMatrix, p2, p_t) + Contract(cMatrix, p_t, p2));
    double term3 = 4*(u - t) * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t));
    double term4 = (-2*u - 3*t + 7*mt*mt) * (Contract(cMatrix, p_t, p1) + Contract(cMatrix, p_tbar, p2));
    double term5 = (3*t - 9*mt*mt) * (Contract(cMatrix, p1, p_tbar) + Contract(cMatrix, p2, p_t));
    return preFactor * (term1 + term2 + term3 + term4 + term5);
}
double delta_M_tu_Vertex(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double preFactor = (gs*gs*gs*gs) / (6.0 * (t - mt*mt) * (u - mt*mt));
    double contraction = mt*mt * (2 * Contract(cMatrix, p1, p2) - 2 * (Contract(cMatrix, p_t, p_tbar) + Contract(cMatrix, p_tbar, p_t)));
    return preFactor * contraction;
}

// Full LIV correction to matrix element (Sum of all propagator and vertex terms)
double DeltaProduction_Full(double gs, double s, double t, double u, double mt, const TMatrixD& cMatrix, const TLorentzVector& p1, const TLorentzVector& p2, const TLorentzVector& p_t, const TLorentzVector& p_tbar) {
    double delta_prop = delta_M_ss_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                      delta_M_tt_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                      delta_M_uu_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                      delta_M_st_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                      delta_M_su_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                      delta_M_tu_Prop(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar);

    double delta_vertex = delta_M_ss_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                        delta_M_tt_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                        delta_M_uu_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                        delta_M_st_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                        delta_M_su_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar) +
                        delta_M_tu_Vertex(gs, s, t, u, mt, cMatrix, p1, p2, p_t, p_tbar);

    return delta_prop + delta_vertex;
}

// Calculate LIV weight (ratio of LIV to SM matrix elements)
double computeLIVWeight(const TLorentzVector& p1, const TLorentzVector& p2,
                        const TLorentzVector& p_t, const TLorentzVector& p_tbar,
                        const TMatrixD& cMatrixLab) {
    double s, t, u;
    calculateMandelstam(p1, p2, p_t, p_tbar, s, t, u);

    // Standard Model matrix element (FULL)
    double M_SM2 = P2gSM_Full(gs, s, t, u, mt);

    // LIV-modified matrix element (FULL correction)
    double delta_M2 = DeltaProduction_Full(gs, s, t, u, mt, cMatrixLab, p1, p2, p_t, p_tbar);
    double M_LIV2 = M_SM2 + delta_M2;

    return M_LIV2 / M_SM2;  // Weight = σ_LIV/σ_SM
}

// Coordinate transformation functions
TMatrixD computeRotationMatrix(double siderealTime, double chi) {
    TMatrixD R(4, 4);
    R.Zero();
    
    double phase = siderealTime;
    R(0, 0) = cos(chi) * cos(phase);
    R(0, 1) = cos(chi) * sin(phase);
    R(0, 2) = -sin(chi);
    R(1, 0) = -sin(phase);
    R(1, 1) = cos(phase);
    R(1, 2) = 0.0;
    R(2, 0) = sin(chi) * cos(phase);
    R(2, 1) = sin(chi) * sin(phase);
    R(2, 2) = cos(chi);
    R(3, 3) = 1.0;  // Time component
    
    return R;
}

TMatrixD rotateCmuNu(const TMatrixD& cSun, const TMatrixD& R) {
    if (cSun.GetNrows() != 4 || cSun.GetNcols() != 4 || 
        R.GetNrows() != 4 || R.GetNcols() != 4) {
        std::cerr << "Error: Matrices must be 4x4" << std::endl;
        return TMatrixD(4,4);
    }

    TMatrixD RT(R);
    RT.Transpose(R);
    TMatrixD cLab(4, 4);
    cLab = R * cSun * RT;
    return cLab;
}

//-----------------------//
//  5. MAIN ANALYSIS FUNCTION //  
//-----------------------//
void CombinedSMEAnalysis() {
    // 4.1 Initialize Delphes and I/O
    gSystem->Load("libDelphes");
    TFile* inputFile = TFile::Open("output_root_file.root", "READ");
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error: Could not open input file!" << std::endl;
        return;
    }

    // Get Delphes tree
    TTree* tree = (TTree*)inputFile->Get("Delphes");
    if (!tree) {
        std::cerr << "Error: Could not find Delphes tree!" << std::endl;
        inputFile->Close();
        return;
    }

    // 4.2 Configure SME parameters
    TMatrixD cMatrixSun(4, 4);
    cMatrixSun.Zero();
    // Set a non-zero coefficient for testing
    cMatrixSun(0, 0) = -0.1;  // Example LIV coefficient c_{TT}
    cMatrixSun(1, 1) = 0.1; // To Keep it traceless i use -
    //cMatrixSun(2, 2) = 0.0;
    //cMatrixSun(3, 3) = 0.0; //c_{TT}
    
    std::cout << "cMatrixSun set to:" << std::endl;
    cMatrixSun.Print();
    
    
    // 4.3 Process events
    TFile* outputFile = new TFile("output_LIV.root", "RECREATE");
    TTree* outTree = tree->CloneTree(0);
    
    double eventWeight, siderealTime;
    outTree->Branch("LIVWeight", &eventWeight, "LIVWeight/D");
    outTree->Branch("SiderealTime", &siderealTime, "SiderealTime/D");

    TClonesArray* particles = nullptr;
    tree->SetBranchAddress("Particle", &particles);
    TRandom3 randGen(12345);

    // Event loop
    for (Long64_t i = 0; i < tree->GetEntries(); i++) {
        tree->GetEntry(i);
        TLorentzVector *top = nullptr, *antitop = nullptr, *p1 = nullptr, *p2 = nullptr;

        // Particle identification
        for (int j = 0; j < particles->GetEntries(); j++) {
            GenParticle* p = (GenParticle*)particles->At(j);
            if (!p) continue;
            
            if (p->PID == 6) top = new TLorentzVector(p->Px, p->Py, p->Pz, p->E);
            else if (p->PID == -6) antitop = new TLorentzVector(p->Px, p->Py, p->Pz, p->E);
            else if (!p1) p1 = new TLorentzVector(p->Px, p->Py, p->Pz, p->E);
            else if (!p2) p2 = new TLorentzVector(p->Px, p->Py, p->Pz, p->E);
        }

        if (top && antitop && p1 && p2) {
            // Coordinate transformation
            siderealTime = 2 * M_PI * randGen.Rndm();
            TMatrixD R = computeRotationMatrix(siderealTime, chi);
            TMatrixD cLab = rotateCmuNu(cMatrixSun, R);

            // Build lab-frame coefficient matrix
            TMatrixD cMatrixLab(4, 4);
            cMatrixLab.Zero();
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    cMatrixLab(mu, nu) = cLab(mu, nu);
                }
            }

            // Calculate LIV weight
            eventWeight = computeLIVWeight(*p1, *p2, *top, *antitop, cMatrixLab);

            if (i < 3) {  // Debug print for first 3 events
                double s, t, u;
                calculateMandelstam(*p1, *p2, *top, *antitop, s, t, u);
                std::cout << "--- Event " << i << " ---" << std::endl;
                std::cout << "Sidereal Time: " << siderealTime << " rad" << std::endl;
                std::cout << "Mandelstam: s=" << s << ", t=" << t << ", u=" << u << std::endl;
                std::cout << "SM ME^2: " << P2gSM_Full(gs, s, t, u, mt) << std::endl;
                std::cout << "LIV Weight: " << eventWeight << std::endl;
            }
            
            outTree->Fill();
        }

        // Clean up
        if (top) delete top;
        if (antitop) delete antitop;
        if (p1) delete p1;
        if (p2) delete p2;
    }

    // 4.4 Save results
    outTree->Write();
    outputFile->Close();
    inputFile->Close();

    // 4.5 Generate plots
    TFile* livFile = TFile::Open("output_LIV.root", "READ");
    if (!livFile || livFile->IsZombie()) {
        std::cerr << "Error: Could not open LIV output file!" << std::endl;
        return;
    }
    TTree* livTree = (TTree*)livFile->Get("Delphes");
    GeneratePlots(livTree);
    livFile->Close();
}

//-----------------------//
//  6. PLOTTING FUNCTIONS //
//-----------------------//
void GeneratePlots(TTree* livTree) {
    // Create histograms
    TH1D* hLIVCrossSection = new TH1D("hLIVCrossSection", 
        "Sum of weights in bin", 
        24, 0, 24);
    
    TH1D* hSMCrossSection = new TH1D("hSMCrossSection", 
        "No. of events Vs sideral time;Sidereal Time [hours];Events", 
        24, 0, 24); //#sigma (arb. units) in place of events

    // Fill histograms
    double eventWeight, siderealTimeRad;
    livTree->SetBranchAddress("LIVWeight", &eventWeight);
    livTree->SetBranchAddress("SiderealTime", &siderealTimeRad);

    Long64_t totalEvents = livTree->GetEntries();


    // DEBUG: PRINT THE KEY VALUES
    std::cout << "=== DEBUG INFORMATION ===" << std::endl;
    std::cout << "Total events (GetEntries()): " << totalEvents << std::endl;
    std::cout << "Events per hour (GetEntries()/24.0): " << totalEvents / 24.0 << std::endl;
    std::cout << "=========================" << std::endl;


    for (Long64_t i = 0; i < totalEvents; i++) {
        livTree->GetEntry(i);
        double siderealHours = siderealTimeRad * (12.0 / M_PI);
        hLIVCrossSection->Fill(siderealHours, eventWeight);
    }

    // Create SM reference
    for (int i = 1; i <= 24; i++) {
        hSMCrossSection->SetBinContent(i, hLIVCrossSection->GetEntries() / 24.0);
    }




    // Plot 1: Comparison
    TCanvas* c1 = new TCanvas("c1", "LIV vs SM Cross Section", 800, 600);
    c1->SetGrid();
    
    // Remove horizontal error bars globally
    gStyle->SetErrorX(0);
    
    hSMCrossSection->SetLineColor(kRed);
    hSMCrossSection->SetLineWidth(2);
    hLIVCrossSection->SetLineColor(kBlue);
    hLIVCrossSection->SetLineWidth(2);
    hLIVCrossSection->SetMarkerStyle(20);
    hLIVCrossSection->SetMarkerColor(kBlue);
    
    hSMCrossSection->Draw("HIST");
    hLIVCrossSection->Draw("SAME E1");  // Vertical errors only
    
    TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
    leg->AddEntry(hSMCrossSection, "SM", "l");
    leg->AddEntry(hLIVCrossSection, "LIV", "lep");
    leg->Draw();
    c1->SaveAs("LIV_vs_SM.png");

    // Plot 2: Ratio
    TCanvas* c2 = new TCanvas("c2", "Ratio: LIV/SM", 800, 600);
    TH1D* hRatio = (TH1D*)hLIVCrossSection->Clone("hRatio");
    hRatio->Divide(hSMCrossSection);
    hRatio->SetTitle("Ratio: LIV/SM;Sidereal Time [hours];#sigma_{LIV}/#sigma_{SM}");
    hRatio->SetLineColor(kBlack);
    hRatio->SetMarkerStyle(20);
    hRatio->Draw("E1");  // Vertical errors only
    
    // Add reference line at 1.0
    TLine* line = new TLine(0, 1.0, 24, 1.0);
    line->SetLineColor(kRed);
    line->SetLineStyle(2);
    line->Draw("SAME");
    
    c2->SaveAs("LIV_SM_Ratio.png");

    // Clean up
    delete c1;
    delete c2;
    delete hRatio;
    delete hSMCrossSection;
    delete hLIVCrossSection;
    delete line;
    delete leg;
}
