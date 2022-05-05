// =============================================================================
//  CADET
//  
//  Copyright © 2008-2021: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================
//TODO: delete iostream
#include <iostream>
/**
 * @file 
 * Defines the general rate model (GRM).
 */

#ifndef LIBCADET_GENERALRATEMODELDG_HPP_
#define LIBCADET_GENERALRATEMODELDG_HPP_

#include "model/UnitOperationBase.hpp"
#include "cadet/StrongTypes.hpp"
#include "cadet/SolutionExporter.hpp"
#include "model/parts/ConvectionDispersionOperator.hpp"
#include "AutoDiff.hpp"
#include "linalg/BandedEigenSparseRowIterator.hpp"
#include "linalg/SparseMatrix.hpp"
#include "linalg/BandMatrix.hpp"
#include "linalg/Gmres.hpp"
#include "Memory.hpp"
#include "model/ModelUtils.hpp"
#include "ParameterMultiplexing.hpp"

#include <Eigen/Dense> // use LA lib Eigen for Matrix operations
#include <Eigen/Sparse>
#include <array>
#include <vector>
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "Benchmark.hpp"

using namespace Eigen;

namespace cadet
{

namespace model
{

namespace parts
{
	namespace cell
	{
		struct CellParameters;
	}
}

class IDynamicReactionModel;
class IParameterDependence;

/**
 * @brief General rate model of liquid column chromatography
 * @details See @cite Guiochon2006, @cite Gu1995, @cite Felinger2004
 * 
 * @f[\begin{align}
	\frac{\partial c_i}{\partial t} &= - u \frac{\partial c_i}{\partial z} + D_{\text{ax},i} \frac{\partial^2 c_i}{\partial z^2} - \frac{1 - \varepsilon_c}{\varepsilon_c} \frac{3}{r_p} j_{f,i} \\
	\frac{\partial c_{p,i}}{\partial t} + \frac{1 - \varepsilon_p}{\varepsilon_p} \frac{\partial q_{i}}{\partial t} &= D_{p,i} \left( \frac{\partial^2 c_{p,i}}{\partial r^2} + \frac{2}{r} \frac{\partial c_{p,i}}{\partial r} \right) + D_{s,i} \frac{1 - \varepsilon_p}{\varepsilon_p} \left( \frac{\partial^2 q_{i}}{\partial r^2} + \frac{2}{r} \frac{\partial q_{i}}{\partial r} \right) \\
	a \frac{\partial q_i}{\partial t} &= f_{\text{iso}}(c_p, q)
\end{align} @f]
@f[ \begin{align}
	j_{f,i} = k_{f,i} \left( c_i - c_{p,i} \left(\cdot, \cdot, r_p\right)\right)
\end{align} @f]
 * Danckwerts boundary conditions (see @cite Danckwerts1953)
@f[ \begin{align}
u c_{\text{in},i}(t) &= u c_i(t,0) - D_{\text{ax},i} \frac{\partial c_i}{\partial z}(t,0) \\
\frac{\partial c_i}{\partial z}(t,L) &= 0 \\
\varepsilon_p D_{p,i} \frac{\partial c_{p,i}}{\partial r}(\cdot, \cdot, r_p) + (1-\varepsilon_p) D_{s,i} \frac{\partial q_{i}}{\partial r}(\cdot, \cdot, r_p) &= j_{f,i} \\
\frac{\partial c_{p,i}}{\partial r}(\cdot, \cdot, 0) &= 0
\end{align} @f]
 * Methods are described in @cite VonLieres2010a (WENO, linear solver), @cite Puttmann2013 @cite Puttmann2016 (forward sensitivities, AD, band compression)
 */
class GeneralRateModelDG : public UnitOperationBase
{
public:

	GeneralRateModelDG(UnitOpIdx unitOpIdx);
	virtual ~GeneralRateModelDG() CADET_NOEXCEPT;

	virtual unsigned int numDofs() const CADET_NOEXCEPT;
	virtual unsigned int numPureDofs() const CADET_NOEXCEPT;
	virtual bool usesAD() const CADET_NOEXCEPT;
	virtual unsigned int requiredADdirs() const CADET_NOEXCEPT;

	virtual UnitOpIdx unitOperationId() const CADET_NOEXCEPT { return _unitOpIdx; }
	virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
	virtual void setFlowRates(active const* in, active const* out) CADET_NOEXCEPT;
	virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
	virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
	virtual bool canAccumulate() const CADET_NOEXCEPT { return false; }

	static const char* identifier() { return "GENERAL_RATE_MODEL_DG"; }
	virtual const char* unitOperationName() const CADET_NOEXCEPT { return identifier(); }

	virtual bool configureModelDiscretization(IParameterProvider& paramProvider, IConfigHelper& helper);
	virtual bool configure(IParameterProvider& paramProvider);
	virtual void notifyDiscontinuousSectionTransition(double t, unsigned int secIdx, const ConstSimulationState& simState, const AdJacobianParams& adJac);

	virtual void useAnalyticJacobian(const bool analyticJac);

	virtual void reportSolution(ISolutionRecorder& recorder, double const* const solution) const;
	virtual void reportSolutionStructure(ISolutionRecorder& recorder) const;

	virtual int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, util::ThreadLocalStorage& threadLocalMem);

	virtual int residualWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);
	virtual int residualSensFwdAdOnly(const SimulationTime& simTime, const ConstSimulationState& simState, active* const adRes, util::ThreadLocalStorage& threadLocalMem);
	virtual int residualSensFwdWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem);

	virtual int residualSensFwdCombine(const SimulationTime& simTime, const ConstSimulationState& simState, 
		const std::vector<const double*>& yS, const std::vector<const double*>& ySdot, const std::vector<double*>& resS, active const* adRes, 
		double* const tmp1, double* const tmp2, double* const tmp3);

	virtual int linearSolve(double t, double alpha, double tol, double* const rhs, double const* const weight,
		const ConstSimulationState& simState);

	virtual void prepareADvectors(const AdJacobianParams& adJac) const;

	virtual void applyInitialCondition(const SimulationState& simState) const;
	virtual void readInitialCondition(IParameterProvider& paramProvider);

	virtual void consistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
	virtual void consistentInitialTimeDerivative(const SimulationTime& simTime, double const* vecStateY, double* const vecStateYdot, util::ThreadLocalStorage& threadLocalMem);

	virtual void initializeSensitivityStates(const std::vector<double*>& vecSensY) const;
	virtual void consistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);

	virtual void leanConsistentInitialState(const SimulationTime& simTime, double* const vecStateY, const AdJacobianParams& adJac, double errorTol, util::ThreadLocalStorage& threadLocalMem);
	virtual void leanConsistentInitialTimeDerivative(double t, double const* const vecStateY, double* const vecStateYdot, double* const res, util::ThreadLocalStorage& threadLocalMem);

	virtual void leanConsistentInitialSensitivity(const SimulationTime& simTime, const ConstSimulationState& simState,
		std::vector<double*>& vecSensY, std::vector<double*>& vecSensYdot, active const* const adRes, util::ThreadLocalStorage& threadLocalMem);

	virtual bool hasInlet() const CADET_NOEXCEPT { return true; }
	virtual bool hasOutlet() const CADET_NOEXCEPT { return true; }

	virtual unsigned int localOutletComponentIndex(unsigned int port) const CADET_NOEXCEPT;
	virtual unsigned int localOutletComponentStride(unsigned int port) const CADET_NOEXCEPT;
	virtual unsigned int localInletComponentIndex(unsigned int port) const CADET_NOEXCEPT;
	virtual unsigned int localInletComponentStride(unsigned int port) const CADET_NOEXCEPT;

	virtual void setExternalFunctions(IExternalFunction** extFuns, unsigned int size);
	virtual void setSectionTimes(double const* secTimes, bool const* secContinuity, unsigned int nSections) { }

	virtual void expandErrorTol(double const* errorSpec, unsigned int errorSpecSize, double* expandOut);

	virtual void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double alpha, double beta, double* ret);
	virtual void multiplyWithDerivativeJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* sDot, double* ret);

	inline void multiplyWithJacobian(const SimulationTime& simTime, const ConstSimulationState& simState, double const* yS, double* ret)
	{
		multiplyWithJacobian(simTime, simState, yS, 1.0, 0.0, ret);
	}

	virtual bool setParameter(const ParameterId& pId, double value);
	virtual bool setParameter(const ParameterId& pId, int value);
	virtual bool setParameter(const ParameterId& pId, bool value);
	virtual bool setSensitiveParameter(const ParameterId& pId, unsigned int adDirection, double adValue);
	virtual void setSensitiveParameterValue(const ParameterId& id, double value);

	virtual std::unordered_map<ParameterId, double> getAllParameterValues() const;
	virtual double getParameterDouble(const ParameterId& pId) const;
	virtual bool hasParameter(const ParameterId& pId) const;

	virtual unsigned int threadLocalMemorySize() const CADET_NOEXCEPT;

#ifdef CADET_BENCHMARK_MODE
	virtual std::vector<double> benchmarkTimings() const
	{
		return std::vector<double>({
			static_cast<double>(numDofs()),
			_timerResidual.totalElapsedTime(),
			_timerResidualPar.totalElapsedTime(),
			_timerResidualSens.totalElapsedTime(),
			_timerResidualSensPar.totalElapsedTime(),
			_timerJacobianPar.totalElapsedTime(),
			_timerConsistentInit.totalElapsedTime(),
			_timerConsistentInitPar.totalElapsedTime(),
			_timerLinearSolve.totalElapsedTime(),
			_timerFactorize.totalElapsedTime(),
			_timerFactorizePar.totalElapsedTime(),
			_timerMatVec.totalElapsedTime(),
			_timerGmres.totalElapsedTime(),
			static_cast<double>(_gmres.numIterations())
		});
	}

	virtual char const* const* benchmarkDescriptions() const
	{
		static const char* const desc[] = {
			"DOFs",
			"Residual",
			"ResidualPar",
			"ResidualSens",
			"ResidualSensPar",
			"JacobianPar",
			"ConsistentInit",
			"ConsistentInitPar",
			"LinearSolve",
			"Factorize",
			"FactorizePar",
			"MatVec",
			"Gmres",
			"NumGMRESIter"
		};
		return desc;
	}
#endif

protected:

	class Indexer;

	int residual(const SimulationTime& simTime, const ConstSimulationState& simState, double* const res, const AdJacobianParams& adJac, util::ThreadLocalStorage& threadLocalMem, bool updateJacobian, bool paramSensitivity);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualImpl(double t, unsigned int secIdx, StateType const* const y, double const* const yDot, ResidualType* const res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualBulk(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType, bool wantJac>
	int residualParticle(double t, unsigned int parType, unsigned int colCell, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res, util::ThreadLocalStorage& threadLocalMem);

	template <typename StateType, typename ResidualType, typename ParamType>
	int residualFlux(double t, unsigned int secIdx, StateType const* y, double const* yDot, ResidualType* res);

	void assembleOffdiagJac(double t, unsigned int secIdx, double const* vecStateY);
	void assembleOffdiagJacFluxParticle(double t, unsigned int secIdx, double const* vecStateY);
	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

	void assembleDiscretizedBulkJacobian(double alpha, Indexer idxr);
	int schurComplementMatrixVector(double const* x, double* z) const;
	void assembleDiscretizedJacobianParticleBlock(unsigned int parType, unsigned int pblk, double alpha, const Indexer& idxr);
	//@TODO?
	void setEquidistantRadialDisc(unsigned int parType);
	void setEquivolumeRadialDisc(unsigned int parType);
	void setUserdefinedRadialDisc(unsigned int parType);
	void updateRadialDisc();

	void addTimeDerivativeToJacobianParticleShell(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, unsigned int parType);
	void solveForFluxes(double* const vecState, const Indexer& idxr) const;
	
	unsigned int numAdDirsForJacobian() const CADET_NOEXCEPT;

	int multiplexInitialConditions(const cadet::ParameterId& pId, unsigned int adDirection, double adValue);
	int multiplexInitialConditions(const cadet::ParameterId& pId, double val, bool checkSens);

	void clearParDepSurfDiffusion();

	parts::cell::CellParameters makeCellResidualParams(unsigned int parType, int const* qsReaction) const;

#ifdef CADET_CHECK_ANALYTIC_JACOBIAN
	void checkAnalyticJacobianAgainstAd(active const* const adRes, unsigned int adDirOffset) const;
#endif

	struct Discretization
	{
		unsigned int nComp; //!< Number of components
		unsigned int nCol; //!< Number of column cells
		unsigned int polyDeg; //!< polynomial degree of column elements
		unsigned int nNodes; //!< Number of nodes per column cell
		unsigned int nPoints; //!< Number of discrete column Points
		unsigned int modal;	//!< bool switch: 1 for modal basis, 0 for nodal basis
		unsigned int nParType; //!< Number of particle types
		unsigned int* nParCell; //!< Array with number of radial cells in each particle type
		unsigned int* nParPointsBeforeType; //!< Array with total number of radial points before a particle type (cumulative sum of nParPoints), additional last element contains total number of particle shells
		unsigned int* parPolyDeg; //!< polynomial degree of particle elements
		unsigned int* nParNode; //!< Array with number of radial nodes per cell in each particle type
		unsigned int* nParPoints; //!< Array with number of radial nodes per cell in each particle type
		unsigned int* parModal;	//!< bool switch: 1 for modal basis, 0 for nodal basis for each particle type
		unsigned int* parTypeOffset; //!< Array with offsets (in particle block) to particle type, additional last element contains total number of particle DOFs
		unsigned int* nBound; //!< Array with number of bound states for each component and particle type (particle type major ordering)
		unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase (particle type major ordering)
		unsigned int* strideBound; //!< Total number of bound states for each particle type, additional last element contains total number of bound states for all types
		unsigned int* nBoundBeforeType; //!< Array with number of bound states before a particle type (cumulative sum of strideBound)

		//////////////////////		DG specifics		////////////////////////////////////////////////////#
		//
		//// NOTE: no different Riemann solvers or boundary conditions

		double deltaZ; //!< equidistant column spacing
		double* deltaR; //!< equidistant particle radial spacing for each particle type
		Eigen::VectorXd nodes; //!< Array with positions of nodes in reference element
		Eigen::MatrixXd polyDerM; //!< Array with polynomial derivative Matrix
		Eigen::VectorXd invWeights; //!< Array with weights for numerical quadrature of size nNodes
		Eigen::MatrixXd invMM; //!< dense !INVERSE! mass matrix for modal (exact) integration
		Eigen::VectorXd* parNodes; //!< Array with positions of nodes in radial reference element for each particle
		Eigen::MatrixXd* parPolyDerM; //!< Array with polynomial derivative Matrix for each particle
		Eigen::VectorXd* parInvWeights; //!< Array with weights for numerical quadrature of size nNodes for each particle
		Eigen::MatrixXd* parInvMM; //!< dense !INVERSE! mass matrix for modal (exact) integration for each particle

		// vgl. convDispOperator
		Eigen::VectorXd dispersion; //!< Column dispersion (may be section and component dependent)
		bool _dispersionCompIndep; //!< Determines whether dispersion is component independent
		double velocity; //!< Interstitial velocity (may be section dependent) \f$ u \f$
		double curVelocity;
		double crossSection; //!< Cross section area 
		int curSection; //!< current time section index

		double length_; //!< column length
		double porosity; //!< column porosity

		Eigen::VectorXd g; //!< auxiliary variable g = dc / dx
		Eigen::VectorXd* g_p; //!< auxiliary variable g = dc_p / dr
		Eigen::VectorXd* g_pSum; //!< auxiliary variable g = sum_{k \in p, s_i} dc_k / dr
		Eigen::VectorXd h;
		Eigen::VectorXd surfaceFlux; //!< stores the surface flux values of the bulk phase
		Eigen::VectorXd* surfaceFluxParticle; //!< stores the surface flux values for each particle
		Eigen::Vector4d boundary; //!< stores the boundary values from Danckwert boundary conditions of the bulk phase
		Eigen::Vector4d pBoundary; //!< stores the boundary values from Danckwert boundary conditions of the bulk phase

		std::vector<bool> isKinetic;

		bool newStaticJac; //!< determines wether static analytical jacobian needs to be computed (every section)
		bool* newStaticJacP; //!< determines wether static analytical jacobian needs to be computed (every section)

		/**
		* @brief computes LGL nodes, integration weights, polynomial derivative matrix
		*/
		void initializeDG() {

			nNodes = polyDeg + 1;
			nPoints = nNodes * nCol;
			// Allocate space for DG operators and containers
			// bulk
			nodes.resize(nNodes);
			nodes.setZero();
			invWeights.resize(nNodes);
			invWeights.setZero();
			polyDerM.resize(nNodes, nNodes);
			polyDerM.setZero();
			invMM.resize(nNodes, nNodes);
			invMM.setZero();
			g.resize(nPoints);
			g.setZero();
			h.resize(nPoints);
			h.setZero();
			boundary.setZero();
			surfaceFlux.resize(nCol + 1);
			surfaceFlux.setZero();
			newStaticJac = true;

			// particle
			nParNode = new unsigned int [nParType];
			nParPoints = new unsigned int [nParType];
			g_p = new VectorXd [nParType];
			g_pSum = new VectorXd [nParType];
			surfaceFluxParticle = new VectorXd [nParType];
			parNodes = new VectorXd [nParType];
			parInvWeights = new VectorXd [nParType];
			parPolyDerM = new MatrixXd [nParType];
			parInvMM = new MatrixXd [nParType];
			newStaticJacP = new bool [nParType * nCol];
			for (int parType = 0; parType < nParType; parType++) {
				nParNode[parType] = parPolyDeg[parType] + 1u;
				nParPoints[parType] = nParNode[parType] * nParCell[parType];
				g_p[parType].resize(nParPoints[parType]);
				g_p[parType].setZero();
				g_pSum[parType].resize(nParPoints[parType]);
				g_pSum[parType].setZero();
				surfaceFluxParticle[parType].resize(nParCell[parType] + 1);
				surfaceFluxParticle[parType].setZero();

				parNodes[parType].resize(nParNode[parType]);
				parNodes[parType].setZero();
				parInvWeights[parType].resize(nParNode[parType]);
				parInvWeights[parType].setZero();
				parPolyDerM[parType].resize(nParNode[parType], nParNode[parType]);
				parPolyDerM[parType].setZero();
				parInvMM[parType].resize(nParNode[parType], nParNode[parType]);
				parInvMM[parType].setZero();
				for (int colNode = 0; colNode < nCol; colNode++) {
					newStaticJacP[parType * nCol + colNode] = true;
				}
			}

			// @TODO: make exact, inexact integration switch during calculation possible?
			// compute DG operators for bulk and every particle
			lglNodesWeights(polyDeg, nodes, invWeights);
			invMMatrix(nNodes, nodes);
			derivativeMatrix(polyDeg, polyDerM, nodes);
			for (int parType = 0; parType < nParType; parType++) {
				lglNodesWeights(parPolyDeg[parType], parNodes[parType], parInvWeights[parType]);
				invMMatrix(nParNode[parType], parNodes[parType]);
				derivativeMatrix(parPolyDeg[parType], parPolyDerM[parType], parNodes[parType]);
			}
		
		}

	private:

		/* ===================================================================================
		*   Polynomial Basis operators and auxiliary functions
		* =================================================================================== */

		/**
		* @brief computes the Legendre polynomial L_N and q = L_N+1 - L_N-2 and q' at point x
		* @param [in] polyDeg polynomial degree of spatial Discretization
		* @param [in] x evaluation point
		* @param [in] L <- L(x)
		* @param [in] q <- q(x) = L_N+1 (x) - L_N-2(x)
		* @param [in] qder <- q'(x) = [L_N+1 (x) - L_N-2(x)]'
		*/
		void qAndL(const int _polyDeg, const double x, double& L, double& q, double& qder) {
			// auxiliary variables (Legendre polynomials)
			double L_2 = 1.0;
			double L_1 = x;
			double Lder_2 = 0.0;
			double Lder_1 = 1.0;
			double Lder = 0.0;
			for (double k = 2; k <= _polyDeg; k++) { // note that this function is only called for polyDeg >= 2.
				L = ((2 * k - 1) * x * L_1 - (k - 1) * L_2) / k;
				Lder = Lder_2 + (2 * k - 1) * L_1;
				L_2 = L_1;
				L_1 = L;
				Lder_2 = Lder_1;
				Lder_1 = Lder;
			}
			q = ((2.0 * _polyDeg + 1) * x * L - _polyDeg * L_2) / (_polyDeg + 1.0) - L_2;
			qder = Lder_1 + (2.0 * _polyDeg + 1) * L_1 - Lder_2;
		}

		/**
		 * @brief computes the Legendre-Gauss-Lobatto nodes and (inverse) quadrature weights
		 * @detail inexact LGL-quadrature leads to a diagonal mass matrix (mass lumping), defined by the quadrature weights
		 */
		void lglNodesWeights(const int _polyDeg, Eigen::VectorXd& _nodes, Eigen::VectorXd& _invWeights) {
			// tolerance and max #iterations for Newton iteration
			int nIterations = 10;
			double tolerance = 1e-15;
			// Legendre polynomial and derivative
			double L = 0;
			double q = 0;
			double qder = 0;
			switch (_polyDeg) {
			case 0:
				throw std::invalid_argument("Polynomial degree must be at least 1 !");
				break;
			case 1:
				_nodes[0] = -1;
				_invWeights[0] = 1;
				_nodes[1] = 1;
				_invWeights[1] = 1;
				break;
			default:
				_nodes[0] = -1;
				_nodes[_polyDeg] = 1;
				_invWeights[0] = 2.0 / (_polyDeg * (_polyDeg + 1.0));
				_invWeights[_polyDeg] = _invWeights[0];
				// use symmetrie, only compute half of points and weights
				for (unsigned int j = 1; j <= floor((_polyDeg + 1) / 2) - 1; j++) {
					//  first guess for Newton iteration
					_nodes[j] = -cos(M_PI * (j + 0.25) / _polyDeg - 3 / (8.0 * _polyDeg * M_PI * (j + 0.25)));
					// Newton iteration to find roots of Legendre Polynomial
					for (unsigned int k = 0; k <= nIterations; k++) {
						qAndL(_polyDeg, _nodes[j], L, q, qder);
						_nodes[j] = _nodes[j] - q / qder;
						if (abs(q / qder) <= tolerance * abs(_nodes[j])) {
							break;
						}
					}
					// calculate weights
					qAndL(_polyDeg, _nodes[j], L, q, qder);
					_invWeights[j] = 2.0 / (_polyDeg * (_polyDeg + 1.0) * pow(L, 2.0));
					_nodes[_polyDeg - j] = -_nodes[j]; // copy to second half of points and weights
					_invWeights[_polyDeg - j] = _invWeights[j];
				}
			}
			if (_polyDeg % 2 == 0) { // for even polyDeg we have an odd number of points which include 0.0
				qAndL(_polyDeg, 0.0, L, q, qder);
				_nodes[_polyDeg / 2] = 0;
				_invWeights[_polyDeg / 2] = 2.0 / (_polyDeg * (_polyDeg + 1.0) * pow(L, 2.0));
			}
			// inverse the weights
			_invWeights = _invWeights.cwiseInverse();
		}

		/**
		 * @brief computation of barycentric weights for fast polynomial evaluation
		 * @param [in] baryWeights vector to store barycentric weights. Must already be initialized with ones!
		 */
		void barycentricWeights(Eigen::VectorXd& baryWeights, const Eigen::VectorXd& _nodes, const int _polyDeg) {
			for (unsigned int j = 1; j <= polyDeg; j++) {
				for (unsigned int k = 0; k <= j - 1; k++) {
					baryWeights[k] = baryWeights[k] * (_nodes[k] - _nodes[j]) * 1.0;
					baryWeights[j] = baryWeights[j] * (_nodes[j] - _nodes[k]) * 1.0;
				}
			}
			for (unsigned int j = 0; j <= _polyDeg; j++) {
				baryWeights[j] = 1 / baryWeights[j];
			}
		}

		/**
		 * @brief computation of nodal (lagrange) polynomial derivative matrix
		 */
		void derivativeMatrix(const int _polyDeg, Eigen::MatrixXd& _polyDerM, const Eigen::VectorXd& _nodes) {
			Eigen::VectorXd baryWeights = Eigen::VectorXd::Ones(_polyDeg + 1u);
			barycentricWeights(baryWeights, _nodes, _polyDeg);
			for (unsigned int i = 0; i <= _polyDeg; i++) {
				for (unsigned int j = 0; j <= _polyDeg; j++) {
					if (i != j) {
						_polyDerM(i, j) = baryWeights[j] / (baryWeights[i] * (_nodes[i] - _nodes[j]));
						_polyDerM(i, i) += -_polyDerM(i, j);
					}
				}
			}
		}

		/**
		 * @brief factor to normalize legendre polynomials
		 */
		double orthonFactor(const int _polyDeg) {

			double n = static_cast<double> (_polyDeg);
			// alpha = beta = 0 to get legendre polynomials as special case from jacobi polynomials.
			double a = 0.0;
			double b = 0.0;
			return std::sqrt(((2.0 * n + a + b + 1.0) * std::tgamma(n + 1.0) * std::tgamma(n + a + b + 1.0))
				/ (std::pow(2.0, a + b + 1.0) * std::tgamma(n + a + 1.0) * std::tgamma(n + b + 1.0)));
		}

		/**
		 * @brief calculates the Vandermonde matrix of the normalized legendre polynomials
		 */
		Eigen::MatrixXd getVandermonde_LEGENDRE(const int _nNodes, const Eigen::VectorXd _nodes) {

			Eigen::MatrixXd V(_nNodes, _nNodes);

			double alpha = 0.0;
			double beta = 0.0;

			// degree 0
			V.block(0, 0, _nNodes, 1) = VectorXd::Ones(_nNodes) * orthonFactor(0);
			// degree 1
			for (int node = 0; node < static_cast<int>(_nNodes); node++) {
				V(node, 1) = _nodes[node] * orthonFactor(1);
			}

			for (int deg = 2; deg <= static_cast<int>(_nNodes - 1); deg++) {

				for (int node = 0; node < static_cast<int>(_nNodes); node++) {

					double orthn_1 = orthonFactor(deg) / orthonFactor(deg - 1);
					double orthn_2 = orthonFactor(deg) / orthonFactor(deg - 2);

					double fac_1 = ((2.0 * deg - 1.0) * 2.0 * deg * (2.0 * deg - 2.0) * _nodes[node]) / (2.0 * deg * deg * (2.0 * deg - 2.0));
					double fac_2 = (2.0 * (deg - 1.0) * (deg - 1.0) * 2.0 * deg) / (2.0 * deg * deg * (2.0 * deg - 2.0));

					V(node, deg) = orthn_1 * fac_1 * V(node, deg - 1) - orthn_2 * fac_2 * V(node, deg - 2);

				}

			}

			return V;
		}

		/**
		* @brief calculates mass matrix for exact polynomial integration
		* @detail exact polynomial integration leads to a full mass matrix
		*/
		void invMMatrix(const int _nnodes, const Eigen::VectorXd _nodes) {
			invMM = (getVandermonde_LEGENDRE(_nnodes, _nodes) * (getVandermonde_LEGENDRE(_nnodes, _nodes).transpose()));
		}

	};

	enum class ParticleDiscretizationMode : int
	{
		/**
		 * Equidistant distribution of shell edges
		 */
		Equidistant,

		/**
		 * Volumes of shells are uniform
		 */
		Equivolume,

		/**
		 * Shell edges specified by user
		 */
		UserDefined
	};

	Discretization _disc; //!< Discretization info
	std::vector<bool> _hasSurfaceDiffusion; //!< Determines whether surface diffusion is present in each particle type
//	IExternalFunction* _extFun; //!< External function (owned by library user)

	parts::ConvectionDispersionOperator _convDispOp; //!< Convection dispersion operator for interstitial volume transport
	parts::ConvectionDispersionOperatorBase _convDispOpB; //!< Convection dispersion operator (base) for interstitial volume transport
	IDynamicReactionModel* _dynReactionBulk; //!< Dynamic reactions in the bulk volume

	Eigen::SparseLU<Eigen::SparseMatrix<double>> _bulkSolver; //!< linear solver for the bulk concentration
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double, RowMajor>, Eigen::DiagonalPreconditioner<double>> _bulkSolver;
	Eigen::SparseLU<Eigen::SparseMatrix<double>>* _parSolver; //!< linear solvers for the particle concentrations

	Eigen::SparseMatrix<double, RowMajor> _jacC; //!< Bulk Jacobian (DG)
	Eigen::SparseMatrix<double, RowMajor> _jacCdisc; //!< Bulk Jacobian (DG) with time derivatove from BDF method

	Eigen::SparseMatrix<double, RowMajor>* _jacP; //!< Particle jacobian diagonal blocks (all of them)
	Eigen::SparseMatrix<double, RowMajor>* _jacPdisc; //!< Particle jacobian diagonal blocks (all of them) with time derivatives from BDF method
	//linalg::BandMatrix* _jacP; //!< Particle jacobian diagonal blocks (all of them)
	//linalg::FactorizableBandMatrix* _jacPdisc; //!< Particle jacobian diagonal blocks (all of them) with time derivatives from BDF method

	linalg::DoubleSparseMatrix _jacCF; //!< Jacobian block connecting interstitial states and fluxes (interstitial transport equation)
	linalg::DoubleSparseMatrix _jacFC; //!< Jacobian block connecting fluxes and interstitial states (flux equation)
	linalg::DoubleSparseMatrix* _jacPF; //!< Jacobian blocks connecting particle states and fluxes (particle transport boundary condition)
	linalg::DoubleSparseMatrix* _jacFP; //!< Jacobian blocks connecting fluxes and particle states (flux equation)

	Eigen::MatrixXd _jacInlet; //!< Jacobian inlet DOF block matrix connects inlet DOFs to first bulk cells

	active _colPorosity; //!< Column porosity (external porosity) \f$ \varepsilon_c \f$
	std::vector<active> _parRadius; //!< Particle radius \f$ r_p \f$
	bool _singleParRadius;
	std::vector<active> _parCoreRadius; //!< Particle core radius \f$ r_c \f$
	bool _singleParCoreRadius;
	std::vector<active> _parPorosity; //!< Particle porosity (internal porosity) \f$ \varepsilon_p \f$
	bool _singleParPorosity;
	std::vector<active> _parTypeVolFrac; //!< Volume fraction of each particle type
	std::vector<ParticleDiscretizationMode> _parDiscType; //!< Particle discretization mode
	std::vector<double> _parDiscVector; //!< Particle discretization shell edges
	std::vector<double> _parGeomSurfToVol; //!< Particle surface to volume ratio factor (i.e., 3.0 for spherical, 2.0 for cylindrical, 1.0 for hexahedral)

	// Vectorial parameters
	std::vector<active> _filmDiffusion; //!< Film diffusion coefficient \f$ k_f \f$
	MultiplexMode _filmDiffusionMode;
	std::vector<active> _parDiffusion; //!< Particle diffusion coefficient \f$ D_p \f$
	MultiplexMode _parDiffusionMode;
	std::vector<active> _parSurfDiffusion; //!< Particle surface diffusion coefficient \f$ D_s \f$
	MultiplexMode _parSurfDiffusionMode;
	std::vector<active> _poreAccessFactor; //!< Pore accessibility factor \f$ F_{\text{acc}} \f$
	MultiplexMode _poreAccessFactorMode;
	std::vector<IParameterDependence*> _parDepSurfDiffusion; //!< Parameter dependencies for particle surface diffusion
	bool _singleParDepSurfDiffusion; //!< Determines whether a single parameter dependence for particle surface diffusion is used
	bool _hasParDepSurfDiffusion; //!< Determines whether particle surface diffusion parameter dependencies are present

	bool _axiallyConstantParTypeVolFrac; //!< Determines whether particle type volume fraction is homogeneous across axial coordinate
	bool _analyticJac; //!< Determines whether AD or analytic Jacobians are used
	unsigned int _jacobianAdDirs; //!< Number of AD seed vectors required for Jacobian computation

	std::vector<active> _parCellSize; //!< Particle node / shell size
	std::vector<active> _parCenterRadius; //!< Particle node-centered position for each particle node
	std::vector<active> _parOuterSurfAreaPerVolume; //!< Particle shell outer sphere surface to volume ratio
	std::vector<active> _parInnerSurfAreaPerVolume; //!< Particle shell inner sphere surface to volume ratio

	ArrayPool _discParFlux; //!< Storage for discretized @f$ k_f @f$ value

	bool _factorizeJacobian; //!< Determines whether the Jacobian needs to be factorized
	double* _tempState; //!< Temporary storage with the size of the state vector or larger if binding models require it
	linalg::Gmres _gmres; //!< GMRES algorithm for the Schur-complement in linearSolve()
	double _schurSafety; //!< Safety factor for Schur-complement solution
	int _colParBoundaryOrder; //!< Order of the bulk-particle boundary discretization

	std::vector<active> _initC; //!< Liquid bulk phase initial conditions
	std::vector<active> _initCp; //!< Liquid particle phase initial conditions
	std::vector<active> _initQ; //!< Solid phase initial conditions
	std::vector<double> _initState; //!< Initial conditions for state vector if given
	std::vector<double> _initStateDot; //!< Initial conditions for time derivative

	BENCH_TIMER(_timerResidual)
	BENCH_TIMER(_timerResidualPar)
	BENCH_TIMER(_timerResidualSens)
	BENCH_TIMER(_timerResidualSensPar)
	BENCH_TIMER(_timerJacobianPar)
	BENCH_TIMER(_timerConsistentInit)
	BENCH_TIMER(_timerConsistentInitPar)
	BENCH_TIMER(_timerLinearSolve)
	BENCH_TIMER(_timerFactorize)
	BENCH_TIMER(_timerFactorizePar)
	BENCH_TIMER(_timerMatVec)
	BENCH_TIMER(_timerGmres)

	// Wrapper for calling the corresponding function in GeneralRateModelDG class
	friend int schurComplementMultiplierGRM_DG(void* userData, double const* x, double* z);

	class Indexer
	{
	public:
		Indexer(const Discretization& disc) : _disc(disc) { }

		// Strides
		inline int strideColNode() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
		inline int strideColCell() const CADET_NOEXCEPT { return static_cast<int>(_disc.nNodes * strideColNode()); }
		inline int strideColComp() const CADET_NOEXCEPT { return 1; }

		inline int strideParComp() const CADET_NOEXCEPT { return 1; }
		inline int strideParLiquid() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
		inline int strideParBound(int parType) const CADET_NOEXCEPT { return static_cast<int>(_disc.strideBound[parType]); }
		inline int strideParShell(int parType) const CADET_NOEXCEPT { return strideParLiquid() + strideParBound(parType); }
		inline int strideParBlock(int parType) const CADET_NOEXCEPT { return static_cast<int>(_disc.nParNode[parType]) * strideParShell(parType); }

		inline int strideFluxNode() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp) * static_cast<int>(_disc.nParType); }
		// @TODO: needed for non-bulk related FV implementation? else delete strideFluxCell
		inline int strideFluxCell() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp) * static_cast<int>(_disc.nParType); }
		inline int strideFluxParType() const CADET_NOEXCEPT { return static_cast<int>(_disc.nComp); }
		inline int strideFluxComp() const CADET_NOEXCEPT { return 1; }

		// Offsets
		inline int offsetC() const CADET_NOEXCEPT { return _disc.nComp; }
		inline int offsetCp() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints + offsetC(); }
		inline int offsetCp(ParticleTypeIndex pti) const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[pti.value]; }
		inline int offsetCp(ParticleTypeIndex pti, ParticleIndex pi) const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[pti.value] + strideParBlock(pti.value) * pi.value; }
		inline int offsetJf() const CADET_NOEXCEPT { return offsetCp() + _disc.parTypeOffset[_disc.nParType]; }
		inline int offsetJf(ParticleTypeIndex pti) const CADET_NOEXCEPT { return offsetJf() + pti.value * _disc.nPoints * _disc.nComp; }
		inline int offsetBoundComp(ParticleTypeIndex pti, ComponentIndex comp) const CADET_NOEXCEPT { return _disc.boundOffset[pti.value * _disc.nComp + comp.value]; }

		// Return pointer to first element of state variable in state vector
		template <typename real_t> inline real_t* c(real_t* const data) const { return data + offsetC(); }
		template <typename real_t> inline real_t const* c(real_t const* const data) const { return data + offsetC(); }

		template <typename real_t> inline real_t* cp(real_t* const data) const { return data + offsetCp(); }
		template <typename real_t> inline real_t const* cp(real_t const* const data) const { return data + offsetCp(); }

		template <typename real_t> inline real_t* q(real_t* const data) const { return data + offsetCp() + strideParLiquid(); }
		template <typename real_t> inline real_t const* q(real_t const* const data) const { return data + offsetCp() + strideParLiquid(); }

		template <typename real_t> inline real_t* jf(real_t* const data) const { return data + offsetJf(); }
		template <typename real_t> inline real_t const* jf(real_t const* const data) const { return data + offsetJf(); }

		// Return specific variable in state vector
		template <typename real_t> inline real_t& c(real_t* const data, unsigned int point, unsigned int comp) const { return data[offsetC() + comp + point * strideColNode()]; }
		template <typename real_t> inline const real_t& c(real_t const* const data, unsigned int point, unsigned int comp) const { return data[offsetC() + comp + point * strideColNode()]; }

	protected:
		const Discretization& _disc;
	};

	class Exporter : public ISolutionExporter
	{
	public:

		Exporter(const Discretization& disc, const GeneralRateModelDG& model, double const* data) : _disc(disc), _idx(disc), _model(model), _data(data) { }
		Exporter(const Discretization&& disc, const GeneralRateModelDG& model, double const* data) = delete;

		virtual bool hasParticleFlux() const CADET_NOEXCEPT { return true; }
		virtual bool hasParticleMobilePhase() const CADET_NOEXCEPT { return true; }
		virtual bool hasSolidPhase() const CADET_NOEXCEPT { return _disc.strideBound[_disc.nParType] > 0; }
		virtual bool hasVolume() const CADET_NOEXCEPT { return false; }

		virtual unsigned int numComponents() const CADET_NOEXCEPT { return _disc.nComp; }
		// @TODO ? actually we need number of axial discrete points here, not number of axial cells, so change name !
		virtual unsigned int numAxialCells() const CADET_NOEXCEPT { return _disc.nPoints; }
		virtual unsigned int numRadialCells() const CADET_NOEXCEPT { return 0; }
		virtual unsigned int numInletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numOutletPorts() const CADET_NOEXCEPT { return 1; }
		virtual unsigned int numParticleTypes() const CADET_NOEXCEPT { return _disc.nParType; }
		virtual unsigned int numParticleShells(unsigned int parType) const CADET_NOEXCEPT { return _disc.nParPoints[parType]; }
		virtual unsigned int numBoundStates(unsigned int parType) const CADET_NOEXCEPT { return _disc.strideBound[parType]; }
		virtual unsigned int numBulkDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints; }
		virtual unsigned int numParticleMobilePhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.nPoints * _disc.nParPoints[parType] * _disc.nComp; }
		virtual unsigned int numSolidPhaseDofs(unsigned int parType) const CADET_NOEXCEPT { return _disc.nPoints * _disc.nParPoints[parType] * _disc.strideBound[parType]; }
		virtual unsigned int numFluxDofs() const CADET_NOEXCEPT { return _disc.nComp * _disc.nPoints * _disc.nParType; }
		virtual unsigned int numVolumeDofs() const CADET_NOEXCEPT { return 0; }

		virtual double const* concentration() const { return _idx.c(_data); }
		virtual double const* flux() const { return _idx.jf(_data); }
		virtual double const* particleMobilePhase(unsigned int parType) const { return _data + _idx.offsetCp(ParticleTypeIndex{parType}); }
		virtual double const* solidPhase(unsigned int parType) const { return _data + _idx.offsetCp(ParticleTypeIndex{parType}) + _idx.strideParLiquid(); }
		virtual double const* volume() const { return nullptr; }
		virtual double const* inlet(unsigned int port, unsigned int& stride) const
		{
			stride = _idx.strideColComp();
			return _data;
		}
		virtual double const* outlet(unsigned int port, unsigned int& stride) const
		{
			stride = _idx.strideColComp();
			if (_model._convDispOp.currentVelocity() >= 0)
				return &_idx.c(_data, _disc.nPoints - 1, 0);
			else
				return &_idx.c(_data, 0, 0);
		}

		virtual StateOrdering const* concentrationOrdering(unsigned int& len) const
		{
			len = _concentrationOrdering.size();
			return _concentrationOrdering.data();
		}

		virtual StateOrdering const* fluxOrdering(unsigned int& len) const
		{
			len = _fluxOrdering.size();
			return _fluxOrdering.data();
		}

		virtual StateOrdering const* mobilePhaseOrdering(unsigned int& len) const
		{
			len = _particleOrdering.size();
			return _particleOrdering.data();
		}

		virtual StateOrdering const* solidPhaseOrdering(unsigned int& len) const
		{
			len = _solidOrdering.size();
			return _solidOrdering.data();
		}

		virtual unsigned int bulkMobilePhaseStride() const { return _idx.strideColNode(); }
		virtual unsigned int particleMobilePhaseStride(unsigned int parType) const { return _idx.strideParShell(parType); }
		virtual unsigned int solidPhaseStride(unsigned int parType) const { return _idx.strideParShell(parType); }

		/**
		* @brief calculates the physical axial/column coordinates of the DG discretization with double! interface nodes
		*/
		virtual void axialCoordinates(double* coords) const {
			Eigen::VectorXd x_l = Eigen::VectorXd::LinSpaced(static_cast<int>(_disc.nCol + 1), 0.0, _disc.length_);
			for (unsigned int i = 0; i < _disc.nCol; i++) {
				for (unsigned int j = 0; j < _disc.nNodes; j++) {
					// mapping 
					coords[i * _disc.nNodes + j] = x_l[i] + 0.5 * (_disc.length_ / static_cast<double>(_disc.nCol)) * (1.0 + _disc.nodes[j]);
				}
			}
		}
		virtual void radialCoordinates(double* coords) const { }
		/**
		* @brief calculates the physical radial/particle coordinates of the DG discretization with double! interface nodes
		*/
		virtual void particleCoordinates(unsigned int parType, double* coords) const
		{
			for (unsigned int par = 0; par < _disc.nParPoints[parType]; par++)
				coords[par] = _disc.deltaR[parType] * std::floor(par / _disc.nParNode[parType])
				+ 0.5 * _disc.deltaR[parType] * (1.0 + _disc.parNodes[parType][par % _disc.nParNode[parType]]);;
		}

	protected:
		const Discretization& _disc;
		const Indexer _idx;
		const GeneralRateModelDG& _model;
		double const* const _data;

		const std::array<StateOrdering, 2> _concentrationOrdering = { { StateOrdering::AxialCell, StateOrdering::Component } };
		const std::array<StateOrdering, 4> _particleOrdering = { { StateOrdering::ParticleType, StateOrdering::AxialCell, StateOrdering::ParticleShell, StateOrdering::Component } };
		const std::array<StateOrdering, 5> _solidOrdering = { { StateOrdering::ParticleType, StateOrdering::AxialCell, StateOrdering::ParticleShell, StateOrdering::Component, StateOrdering::BoundState } };
		const std::array<StateOrdering, 3> _fluxOrdering = { { StateOrdering::ParticleType, StateOrdering::AxialCell, StateOrdering::Component } };
	};

	/**
	* @brief sets the current section index and section dependend velocity, dispersion
	*/
	void updateSection(int secIdx) {

		if (cadet_unlikely(_disc.curSection != secIdx)) {

			_disc.curSection = secIdx;
			_disc.newStaticJac = true;
			for (int par = 0; par < _disc.nCol * _disc.nParType; par++) {
				_disc.newStaticJacP[par] = true;
			}

			// update velocity and dispersion
			_disc.velocity = static_cast<double>(_convDispOpB.currentVelocity());
			if (_convDispOpB.dispersionCompIndep())
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					_disc.dispersion[comp] = static_cast<double>(_convDispOpB.currentDispersion(secIdx)[0]);
				}
			else {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					_disc.dispersion[comp] = static_cast<double>(_convDispOpB.currentDispersion(secIdx)[comp]);
				}
			}

			//std::cout << "NEW SECTION: " << _disc.curSection << std::endl;
			//std::cout << "v: " << _disc.velocity << std::endl;
			//std::cout << "D_ax: " << _disc.dispersion << std::endl;
		}
	}

// ==========================================================================================================================================================  //
// ========================================						DG RHS						======================================================  //
// ==========================================================================================================================================================  //

	/**
	* @brief calculates the volume Integral of the auxiliary equation
	* @detail estimates the state derivative = - D * state
	* @param [in] current state vector
	* @param [in] stateDer vector to be changed
	* @param [in] aux true if auxiliary, else main equation
	*/
	void volumeIntegral(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer) {
		// comp-cell-node state vector: use of Eigen lib performance
		for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
			stateDer.segment(Cell * _disc.nNodes, _disc.nNodes)
				-= _disc.polyDerM * state.segment(Cell * _disc.nNodes, _disc.nNodes);
		}
	}
	void parVolumeIntegral(int parType, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer) {
		// comp-cell-node state vector: use of Eigen lib performance
		for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
			stateDer.segment(Cell * _disc.nParNode[parType], _disc.nParNode[parType])
				-= _disc.parPolyDerM[parType] * state.segment(Cell * _disc.nParNode[parType], _disc.nParNode[parType]);
		}
	}

	/*
	* @brief calculates the interface fluxes h* of Convection Dispersion equation
	*/
	void InterfaceFlux(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, const VectorXd& g, unsigned int comp) {

		// component-wise strides
		unsigned int strideCell = _disc.nNodes;
		unsigned int strideNode = 1u;

		// Conv.Disp. flux: h* = h*_conv + h*_disp = numFlux(v c_l, v c_r) + 0.5 sqrt(D_ax) (S_l + S_r)

		// calculate inner interface fluxes
		for (unsigned int Cell = 1; Cell < _disc.nCol; Cell++) {
			// h* = h*_conv + h*_disp
			_disc.surfaceFlux[Cell] // inner interfaces
				= _disc.velocity * (C[Cell * strideCell - strideNode])
				- 0.5 * std::sqrt(_disc.dispersion[comp]) * (g[Cell * strideCell - strideNode] // left cell
					+ g[Cell * strideCell]);
		}

		// boundary fluxes
			// left boundary interface
		_disc.surfaceFlux[0]
			= _disc.velocity * _disc.boundary[0];

		// right boundary interface
		_disc.surfaceFlux[_disc.nCol]
			= _disc.velocity * (C[_disc.nCol * strideCell - strideNode])
			- std::sqrt(_disc.dispersion[comp]) * 0.5 * (g[_disc.nCol * strideCell - strideNode] // last cell last node
				+ _disc.boundary[3]); // right boundary value S
	}

	void InterfaceFluxParticle(int parType, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		unsigned int strideCell, unsigned int strideNode, bool aux) {

		// reset surface flux storage as it is used multiple times
		_disc.surfaceFluxParticle[parType].setZero();

		// numerical flux: state* = 0.5 (state^+ + state^-)

		// calculate inner interface fluxes
		for (unsigned int Cell = 1u; Cell < _disc.nParCell[parType]; Cell++) {
			_disc.surfaceFluxParticle[parType][Cell] // left interfaces
				= 0.5 * (state[Cell * strideCell - strideNode] + // outer/left node
					state[Cell * strideCell]); // inner/right node
		}

		// calculate boundary interface fluxes.
		// Note that inflow boundary conditions are handled in residualFlux()
		if (aux) { // ghost nodes given by state^- := state^+ for auxiliary equation
			_disc.surfaceFluxParticle[parType][0] = state[0];

			_disc.surfaceFluxParticle[parType][_disc.nParCell[parType]] = state[_disc.nParCell[parType] * strideCell - strideNode];
		}
		else { // ghost nodes given by state^- := 0.0 for main equation
			_disc.surfaceFluxParticle[parType][0] = 0.0;// state[0];

			_disc.surfaceFluxParticle[parType][_disc.nParCell[parType]] = 0.0; // state[parDisc.nCells * strideCell - strideNode];
		}
	}

	/**
	* @brief calculates and fills the surface flux values for auxiliary equation
	* @param [in] strideCell component-wise cell stride
	* @param [in] strideNodecomponent-wise node stride
	*/
	void InterfaceFluxAuxiliary(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, unsigned int strideCell, unsigned int strideNode) {

		// Auxiliary flux: c* = 0.5 (c^+ + c^-)

		// calculate inner interface fluxes
		for (unsigned int Cell = 1; Cell < _disc.nCol; Cell++) {
			_disc.surfaceFlux[Cell] // left interfaces
				= 0.5 * (C[Cell * strideCell - strideNode] + // left node
					C[Cell * strideCell]); // right node
		}
		// calculate boundary interface fluxes

		_disc.surfaceFlux[0] // left boundary interface
			= 0.5 * (C[0] + // boundary value
				C[0]); // first cell first node

		_disc.surfaceFlux[(_disc.nCol)] // right boundary interface
			= 0.5 * (C[_disc.nCol * strideCell - strideNode] + // last cell last node
				C[_disc.nCol * strideCell - strideNode]);// // boundary value
	}

	/**
	* @brief calculates the surface Integral, depending on the approach (modal/nodal)
	* @param [in] state relevant state vector
	* @param [in] stateDer state derivative vector the solution is added to
	* @param [in] aux true for auxiliary equation, false for main equation
		surfaceIntegral(cPtr, &(disc.g[0]), disc,&(disc.h[0]), resPtrC, 0, secIdx);
	* @param [in] strideCell component-wise cell stride
	* @param [in] strideNodecomponent-wise node stride
	*/
	void surfaceIntegral(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer, bool aux, unsigned int Comp, unsigned int strideCell, unsigned int strideNode) {

		// calc numerical flux values c* or h* depending on equation switch aux
		(aux == 1) ? InterfaceFluxAuxiliary(C, strideCell, strideNode) : InterfaceFlux(C, _disc.g, Comp);
		if (_disc.modal) { // modal approach -> dense mass matrix
			for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
				// strong surface integral -> M^-1 B [state - state*]
				for (unsigned int Node = 0; Node < _disc.nNodes; Node++) {
					stateDer[Cell * strideCell + Node * strideNode]
						-= _disc.invMM(Node, 0) * (state[Cell * strideCell]
							- _disc.surfaceFlux[Cell])
						- _disc.invMM(Node, _disc.polyDeg) * (state[Cell * strideCell + _disc.polyDeg * strideNode]
							- _disc.surfaceFlux[(Cell + 1u)]);
				}
			}
		}
		else { // nodal approach -> diagonal mass matrix
			for (unsigned int Cell = 0; Cell < _disc.nCol; Cell++) {
				// strong surface integral -> M^-1 B [state - state*]
				stateDer[Cell * strideCell] // first cell node
					-= _disc.invWeights[0] * (state[Cell * strideCell] // first node
						- _disc.surfaceFlux(Cell));
				stateDer[Cell * strideCell + _disc.polyDeg * strideNode] // last cell node
					+= _disc.invWeights[_disc.polyDeg] * (state[Cell * strideCell + _disc.polyDeg * strideNode]
						- _disc.surfaceFlux(Cell + 1u));
			}
		}
	}
	void parSurfaceIntegral(int parType, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer, unsigned int strideCell, unsigned int strideNode, bool aux) {

		// calc numerical flux values
		InterfaceFluxParticle(parType, state, strideCell, strideNode, aux);
		if (_disc.modal) { // modal approach -> dense mass matrix
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
				// strong surface integral -> M^-1 B [state - state*]
				for (unsigned int Node = 0; Node < _disc.nParNode[parType]; Node++) {
					stateDer[Cell * strideCell + Node * strideNode]
						-= _disc.parInvMM[parType](Node, 0) * (state[Cell * strideCell]
							- _disc.surfaceFluxParticle[parType][Cell])
						- _disc.parInvMM[parType](Node, _disc.parPolyDeg[parType]) * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode]
							- _disc.surfaceFluxParticle[parType][(Cell + 1u)]);
				}
			}
		}
		else { // nodal approach -> diagonal mass matrix
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
				// strong surface integral -> M^-1 B [state - state*]
				stateDer[Cell * strideCell] // first cell node
					-= _disc.parInvWeights[parType][0] * (state[Cell * strideCell] // first node
						- _disc.surfaceFluxParticle[parType][Cell]);
				stateDer[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] // last cell node
					+= _disc.parInvWeights[parType][_disc.parPolyDeg[parType]] * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode]
						- _disc.surfaceFluxParticle[parType][Cell + 1u]);
			}
		}
	}

	/**
	* @brief calculates the substitute h = vc - sqrt(D_ax) g(c)
	*/
	void calcH(Eigen::Map<const VectorXd, 0, InnerStride<>>& C, unsigned int Comp) {
		_disc.h = _disc.velocity * C - std::sqrt(_disc.dispersion[Comp]) * _disc.g;
	}

	/**
	* @brief applies the inverse Jacobian of the mapping
	*/
	void applyMapping(Eigen::Map<VectorXd, 0, InnerStride<>>& state) {
		state *= (2.0 / _disc.deltaZ);
	}
	/**
	* @brief applies the inverse Jacobian of the mapping and auxiliary factor -1
	*/
	void applyMapping_Aux(Eigen::Map<VectorXd, 0, InnerStride<>>& state, unsigned int Comp) {
		state *= (-2.0 / _disc.deltaZ) * ((_disc.dispersion[Comp] == 0.0) ? 1.0 : std::sqrt(_disc.dispersion[Comp]));
	}

	void ConvDisp_DG(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& resC, double t, unsigned int Comp) {

		// ===================================//
		// reset cache                        //
		// ===================================//

		resC.setZero();
		_disc.h.setZero();
		_disc.g.setZero();
		_disc.surfaceFlux.setZero();
		// get Map objects of auxiliary variable memory
		Eigen::Map<VectorXd, 0, InnerStride<>> g(&_disc.g[0], _disc.nPoints, InnerStride<>(1));
		Eigen::Map<const VectorXd, 0, InnerStride<>> h(&_disc.h[0], _disc.nPoints, InnerStride<>(1));

		// ======================================//
		// solve auxiliary system g = d c / d x  //
		// ======================================//

		volumeIntegral(C, g); // DG volumne integral in strong form

		surfaceIntegral(C, C, g, 1, Comp, _disc.nNodes, 1u); // surface integral in strong form

		applyMapping_Aux(g, Comp); // inverse mapping from reference space and auxiliary factor

		_disc.surfaceFlux.setZero(); // reset surface flux storage as it is used twice

		// ======================================//
		// solve main equation w_t = d h / d x   //
		// ======================================//

		calcH(C, Comp); // calculate the substitute h(S(c), c) = sqrt(D_ax) g(c) - v c

		volumeIntegral(h, resC); // DG volumne integral in strong form

		calcBoundaryValues(C);// update boundary values including auxiliary variable g

		surfaceIntegral(C, h, resC, 0, Comp, _disc.nNodes, 1u); // DG surface integral in strong form

		applyMapping(resC); // inverse mapping to reference space

	}

	/**
	* @brief computes ghost nodes used to implement Danckwerts boundary conditions of bulk phase
	*/
	void calcBoundaryValues(Eigen::Map<const VectorXd, 0, InnerStride<>>& C) {

		//cache.boundary[0] = c_in -> inlet DOF idas suggestion
		_disc.boundary[1] = C[_disc.nPoints - 1]; // c_r outlet
		_disc.boundary[2] = -_disc.g[0]; // S_l inlet
		_disc.boundary[3] = -_disc.g[_disc.nPoints - 1]; // g_r outlet
	}

	// Particle specific RHS functions
	
	/**
	* @brief solves the auxiliary system g = d conc / d xi
	*/
	void solve_auxiliary_DG(int parType, Eigen::Map<const VectorXd, 0, InnerStride<>>& conc, unsigned int strideCell, unsigned int strideNode) {

		Indexer idxr(_disc);
		Eigen::Map<VectorXd, 0, InnerStride<>> g_p(&_disc.g_p[parType][0], _disc.nParPoints[parType], InnerStride<>(1));
		unsigned int comp = 0; // comp is just a placeholder here

		// =================================================================//
		// solve auxiliary systems g_p + sum g_s= d (c_p sum c_s) / d r     //
		// =================================================================//
		
		// reset surface flux storage as it is used multiple times
		_disc.surfaceFluxParticle[parType].setZero();
		// reset auxiliary g
		g_p.setZero();
		// DG volumne integral: - D c
		parVolumeIntegral(parType, conc, g_p);
		// surface integral: M^-1 B [c - c^*]
		parSurfaceIntegral(parType, conc, g_p, strideCell, strideNode, true);
		// inverse mapping from reference space and auxiliary factor -1
		g_p *= - 2.0 / _disc.deltaR[parType];

	}

	// ==========================================================================================================================================================  //
	// ========================================						DG Jacobian							=========================================================  //
	// ==========================================================================================================================================================  //

	typedef Eigen::Triplet<double> T;

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian
	*/
	void setConvDispJacPattern(Eigen::SparseMatrix<double, RowMajor>& mat) {

		std::vector<T> tripletList;
		// TODO?: convDisp NNZ times two for now, but Convection NNZ < Dispersion NNZ
		tripletList.reserve(2u * calcConvDispNNZ(_disc));

		if (_disc.modal)
			ConvDispModalPattern(tripletList);
		else
			ConvDispNodalPattern(tripletList);

		mat.setFromTriplets(tripletList.begin(), tripletList.end());

	}
	void setParDispJacPattern(int parType, Eigen::SparseMatrix<double, RowMajor>& mat) {

		std::vector<T> tripletList;
		// TODO?: convDisp NNZ times two for now, but Convection NNZ < Dispersion NNZ
		tripletList.reserve(calcParDispNNZ(parType, _disc));

		if (!_disc.parModal[parType])
			calcNodalParticleJacobianPattern(parType, tripletList);
		//else
			//calcModalParticleJacobianPattern(parType, tripletList, mat);

		isothermPattern(parType, tripletList);

		mat.setFromTriplets(tripletList.begin(), tripletList.end());

	}

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian for the nodal DG scheme
	*/
	int ConvDispNodalPattern(std::vector<T>& tripletList) {

		Indexer idx(_disc);

		int sNode = idx.strideColNode();
		int sCell = idx.strideColCell();
		int sComp = idx.strideColComp();
		int offC = 0; // inlet DOFs not included in Jacobian

		unsigned int nNodes = _disc.nNodes;
		unsigned int polyDeg = _disc.polyDeg;
		unsigned int nCells = _disc.nCol;
		unsigned int nComp = _disc.nComp;

		/*======================================================*/
		/*			Define Convection Jacobian Block			*/
		/*======================================================*/

		// Convection block [ d RHS_conv / d c ], also depends on first entry of previous cell

		// special inlet DOF treatment for first cell
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < nNodes; i++) {
				//tripletList.push_back(T(offC + comp * sComp + i * sNode, comp * sComp, 0.0)); // inlet DOFs not included in Jacobian
				for (unsigned int j = 1; j < nNodes + 1; j++) {
					tripletList.push_back(T(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - 1) * sNode,
						0.0));
				}
			}
		}
		for (unsigned int cell = 1; cell < nCells; cell++) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
						// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
						tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
							offC + cell * sCell - sNode + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}

		/*======================================================*/
		/*			Define Dispersion Jacobian Block			*/
		/*======================================================*/

		/*		Inner cell dispersion blocks		*/


		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell

		// insert Blocks to Jacobian inner cells (only for nCells >= 3)
		if (nCells >= 3u) {
			for (unsigned int cell = 1; cell < nCells - 1; cell++) {
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = 0; j < 3 * nNodes; j++) {
							// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry
							tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
								offC + (cell - 1) * sCell + comp * sComp + j * sNode,
								0.0));
						}
					}
				}
			}
		}

		/*				Boundary cell Dispersion blocks			*/

		if (nCells != 1) { // Note: special case nCells = 1 already set by advection block
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = nNodes; j < 3 * nNodes; j++) {
						tripletList.push_back(T(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - nNodes) * sNode,
							0.0));
					}
				}
			}

			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 1 - 1) * sCell + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}

		return 0;
	}

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian for the modal DG scheme
	*/
	int ConvDispModalPattern(std::vector<T>& tripletList) {

		Indexer idx(_disc);

		int sNode = idx.strideColNode();
		int sCell = idx.strideColCell();
		int sComp = idx.strideColComp();
		int offC = 0; // inlet DOFs not included in Jacobian

		unsigned int nNodes = _disc.nNodes;
		unsigned int nCells = _disc.nCol;
		unsigned int nComp = _disc.nComp;

		/*======================================================*/
		/*			Define Convection Jacobian Block			*/
		/*======================================================*/

		// Convection block [ d RHS_conv / d c ], additionally depends on first entry of previous cell
		// special inlet DOF treatment for first cell
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < nNodes; i++) {
				//tripletList.push_back(T(offC + comp * sComp + i * sNode, comp * sComp, 0.0)); // inlet DOFs not included in Jacobian
				for (unsigned int j = 1; j < nNodes + 1; j++) {
					tripletList.push_back(T(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - 1) * sNode,
						0.0));
				}
			}
		}
		for (unsigned int cell = 1; cell < nCells; cell++) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
						// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
						tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
							offC + cell * sCell - sNode + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}

		/*======================================================*/
		/*			Define Dispersion Jacobian Block			*/
		/*======================================================*/

		/* Inner cells */
		if (nCells >= 5u) {
			// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			for (unsigned int cell = 2; cell < nCells - 2; cell++) {
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = 0; j < 3 * nNodes + 2; j++) {
							// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry
							tripletList.push_back(T(offC + cell * sCell + comp * sComp + i * sNode,
								offC + cell * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
								0.0));
						}
					}
				}
			}
		}

		/*		boundary cell neighbours		*/

		// left boundary cell neighbour
		if (nCells >= 4u) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 1; j < 3 * nNodes + 2; j++) {
						// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
						tripletList.push_back(T(offC + nNodes * sNode + comp * sComp + i * sNode,
							offC + comp * sComp + (j - 1) * sNode,
							0.0));
					}
				}
			}
		}
		else if (nCells == 3u) { // special case: only depends on the two neighbouring cells
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 1; j < 3 * nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
						tripletList.push_back(T(offC + nNodes * sNode + comp * sComp + i * sNode,
							offC + comp * sComp + (j - 1) * sNode,
							0.0));
					}
				}
			}
		}
		// right boundary cell neighbour
		if (nCells >= 4u) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 3 * nNodes + 2 - 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
						tripletList.push_back(T(offC + (nCells - 2) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 2) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}
		/*			boundary cells			*/

		// left boundary cell
		unsigned int end = 3u * nNodes + 2u;
		if (nCells == 1u) end = 2u * nNodes + 1u;
		else if (nCells == 2u) end = 3u * nNodes + 1u;
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < nNodes; i++) {
				for (unsigned int j = nNodes + 1; j < end; j++) {
					// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
					// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
					tripletList.push_back(T(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - (nNodes + 1)) * sNode,
						0.0));
				}
			}
		}
		// right boundary cell
		if (nCells >= 3u) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 2 * nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
						tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 1) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}
		else if (nCells == 2u) { // special case for nCells == 2: depends only on left cell
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry.
						tripletList.push_back(T(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 1) * sCell - (nNodes)*sNode + comp * sComp + j * sNode,
							0.0));
					}
				}
			}
		}

		return 0;
	}

	/**
	* @brief analytically calculates the static (per section) bulk jacobian (inlet DOFs included!)
	* @return 1 if jacobain estimation fits the predefined pattern of the jacobian, 0 if not.
	*/
	int calcStaticAnaBulkJacobian(double t, unsigned int secIdx, const double* const y, util::ThreadLocalStorage& threadLocalMem) {

		// not necessary: set _jac to zero but keep pattern ! (.setZero() deletes pattern)
		//double* vPtr = _jac.valuePtr();
		//for (int k = 0; k < _jac.nonZeros(); k++) {
		//	vPtr[k] = 0.0;
		//}

			// DG convection dispersion Jacobian
		if (_disc.modal)
			calcConvDispModalJacobian(_jacC);
		else
			calcConvDispNodalJacobian(_jacC);

		if (!_jacC.isCompressed()) // if matrix lost its compressed storage, the pattern did not fit.
			return 0;

		//MatrixXd mat = _jacC.toDense();
		//for (int i = 0; i < mat.rows(); i++) {
		//	for (int j = 0; j < mat.cols(); j++) {
		//		//if (mat(i, j) != 0) {
		//		//	mat(i, j) = 1.0;
		//		//}
		//	}
		//}
		//std::cout << std::fixed << std::setprecision(4) << "JAC_Bulk\n" << mat << std::endl; // #include <iomanip>

		return 1;
	}

	/**
	* @brief analytically calculates the static (per section) particle jacobian
	* @return 1 if jacobain calculation fits the predefined pattern of the jacobian, 0 if not.
	*/
	int calcStaticAnaParticleJacobian(unsigned int parType, const double* const parDiff, const double* const parSurfDiff, const double* const invBetaP,
		unsigned int colNode, double t, unsigned int secIdx, const double* const y, util::ThreadLocalStorage& threadLocalMem) {

		// not necessary: set _jac to zero but keep pattern ! (.setZero() deletes pattern)
		//double* vPtr = _jac.valuePtr();
		//for (int k = 0; k < _jac.nonZeros(); k++) {
		//	vPtr[k] = 0.0;
		//}

		// DG particle dispersion Jacobian
		if (_disc.parModal[parType])
			throw std::invalid_argument("modal/exact integration Particle Jacobian not implemented yet");
			//calcModalParticleJacobian(parType, _jacP[parType * colNode]);
		else
			calcNodalParticleJacobian(parType, parDiff, parSurfDiff, invBetaP, _jacP[parType * _disc.nPoints + colNode]);

		if (!_jacP[parType * _disc.nPoints + colNode].isCompressed()) // if matrix lost its compressed storage, the calculation did not fit the pre-defined pattern.
			return 0;

		//MatrixXd mat = _jacP[parType * colNode].toDense();
		//for (int i = 0; i < mat.rows(); i++) {
		//	for (int j = 0; j < mat.cols(); j++) {
		//		//if (mat(i, j) != 0) {
		//		//	mat(i, j) = 1.0;
		//		//}
		//	}
		//}
		//std::cout << std::fixed << std::setprecision(4) << "JAC_particle\n" << mat << std::endl; // #include <iomanip>

		return 1;
	}

	/**
	 * @brief calculates the particle dispersion jacobian Pattern of the nodal DG scheme for one particle type and bead
	*/
	void calcNodalParticleJacobianPattern(unsigned int parType, std::vector<T>& tripletList) {

		// (global) strides
		Indexer idxr(_disc);
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParShell(parType);
		unsigned int sNode = idxr.strideParShell(parType);
		unsigned int sComp = 1u;

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < _disc.nParPoints[parType]; i++) {
					for (unsigned int j = 0; j < _disc.nParPoints[parType]; j++) {
						// handle liquid state: self dependency
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						tripletList.push_back(T(comp * sComp + i * sNode,
											    comp * sComp + j * sNode,
							0.0));

						// handle bound states: surface diffusion
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
							// with surface diffusion, the residual of the particle equation depends on all bound states
							// row: add component offset and go node strides from there for each dispersion block entry
							// col: jump over liquid states, add offset to first bound state of current component, add current bound state offset
							//		and go node strides from there for each dispersion block entry
							tripletList.push_back(T(comp * sComp + i * sNode,
													idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode,
								0.0));
							}
						}
					}
					// handle bound states: binding (local dependency on point i)
					for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
						// row:	add current component offset. jump over previous points and liquid concentration, add the offset of the current bound state
						// col: jump over previous points and add entries for all liquid and bound concentrations
						for (int conc = 0; conc < sNode; conc++) {
							tripletList.push_back(T(comp * sComp + i * sNode + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd,
													i * sNode + conc,
								0.0));
						}
					}
				}
			}

		}
		else if (_disc.nParCell[parType] == 2) {
			throw std::invalid_argument("Particle Jacobian only implemented for nCells = 1 for now");

			//dispBlock = MatrixXd::Zero(1, 1);
			//dispBlock = invMap * (_disc.parPolyDerM[parType] * _disc.parPolyDerM[parType] - invMM * B * _disc.parPolyDerM[parType]);


		}
		else {
			throw std::invalid_argument("Particle Jacobian only implemented for nCells = 1 for now");


		}
	}

	/**
	* @brief adds the sparsity pattern of the isotherm Jacobian
	* @detail Independent of the isotherm, all liquid and solid entries (so all entries, the isotherm could theoretically depend on) at a discrete point are set.
	*/
	void isothermPattern(int parType, std::vector<T>& tripletList) {

		Indexer idxr(_disc);

		// loop over all discrete points and solid states and add all liquid plus solid entries at that solid state at that discrete point
		for (unsigned int point = 0; point < _disc.nParPoints[parType]; point++) {
			for (unsigned int solid = 0; solid < _disc.strideBound[parType]; solid++) {
				for (unsigned int conc = 0; conc < _disc.nComp + _disc.strideBound[parType]; conc++) {
					// row:		  jump over previous discrete points and liquid concentration and add the offset of the current bound state
					// column:    jump over previous discrete points. add entries for all liquid and bound concentrations (conc)
					tripletList.push_back(T(idxr.strideParShell(parType) * point + idxr.strideParLiquid() + solid,
											idxr.strideParShell(parType) * point + conc,
						0.0));
				}
			}
		}

	}

	/**
	 * @brief returns particle diffusion coefficients for one component particle liquid and bound concentrations
	*/
	const Eigen::VectorXd getParDiffComp(unsigned int parType, unsigned int comp, const double* const parDiff, const double* const parSurfDiff) {

		Eigen::VectorXd diff = Eigen::VectorXd::Zero(1 + _disc.nBound[parType * _disc.nComp + comp]);

		// ordering of parSurfDiff: bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
		// -> so we need to get the strides in order to extract the component of interest
		unsigned int* nBnd = new unsigned int[_disc.nComp];
		std::copy_n(&_disc.nBound[parType * _disc.nComp], _disc.nComp, &nBnd[0]);
		Eigen::VectorXi stride = Eigen::VectorXi::Zero(_disc.nBound[parType * _disc.nComp + comp] + 1); // last entry is just a placeholder
		for (int i = 0; i < _disc.nBound[parType * _disc.nComp + comp]; i++) {
			for (int _comp = 0; _comp < _disc.nComp; _comp++) {
				if (nBnd[_comp] > 0) {
					(_comp < comp) ? stride[i]++ : stride[i + 1]++;
					nBnd[_comp]--;
				}
			}
			if (i > 0)
				stride[i] += stride[i - 1];
		}

		diff[0] = parDiff[comp];
		for (int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
			diff[bnd + 1] = parSurfDiff[stride[bnd]];
		}
		return diff;
	}

	/**
	 * @brief analytically calculates the particle dispersion jacobian of the nodal DG scheme for one particle type and bead
	*/
	int calcNodalParticleJacobian(unsigned int parType, const double* const parDiff, const double* const parSurfDiff, const double* const invBetaP,
		Eigen::SparseMatrix<double, RowMajor>& jacP) {

		// (global) strides
		Indexer idxr(_disc);
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParShell(parType);
		unsigned int sNode = idxr.strideParShell(parType);
		unsigned int sComp = 1u;
		//
		unsigned int nNodes = _disc.nParNode[parType];

		// blocks to compute jacobian
		Eigen::MatrixXd dispBlock;
		double invMap = (2.0 / _disc.deltaR[parType]);
		Eigen::MatrixXd B = MatrixXd::Zero(_disc.nParNode[parType], _disc.nParNode[parType]);
		B(0, 0) = -1.0; B(_disc.nParNode[parType] - 1, _disc.nParNode[parType] - 1) = 1.0;
		Eigen::MatrixXd invMM = _disc.parInvWeights[parType].asDiagonal();
		Eigen::MatrixXd M_1M_rG;

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			M_1M_rG = MatrixXd::Zero(nNodes, nNodes);
			dispBlock = invMap * (_disc.parPolyDerM[parType] * _disc.parPolyDerM[parType]);

			// compute metric part separately, because of cell dependend left interface coordinate r_L
			double r_L = static_cast<double>(_parRadius[parType]) - 0.0 * _disc.deltaR[parType]; // left boundary of current cell
			for (int node = 0; node < nNodes; node++) {
				M_1M_rG(node, node) = 2.0 / (r_L + (_disc.deltaR[parType] / 2.0) * (_disc.parNodes[parType][node] + 1.0));
			}
			M_1M_rG *= _disc.parPolyDerM[parType];

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				// get bound and liquid diffusion coefficients for current component
				Eigen::VectorXd diff = getParDiffComp(parType, comp, parDiff, parSurfDiff);
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						jacP.coeffRef(comp * sComp + i * sNode,
									  comp * sComp + j * sNode)
							= -(invMap * diff[0])
							* (M_1M_rG(i, j) + dispBlock(i, j)); // dispBlock += invMap* D_p * (M^-1 * M_r * D + invMap * (D * D - M^-1 * B * D))

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								jacP.coeffRef(comp * sComp + i * sNode,
											  idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
									= -(invMap * diff[bnd + 1] * invBetaP[comp])
									* (M_1M_rG(i, j) + dispBlock(i, j)); // dispBlock = 2/D_r* D_p * (M^-1 * M_r * D + invMap * (D * D - M^-1 * B * D))
							}
						}
					}
				}
			}
			//for (int i = 0; i < _disc.nParPoints[parType]; i++) {
			//	jacP.coeffRef(comp * sComp + i * sNode,
			//				  comp * sComp + i * sNode)
			//		= 0.0;
			//}
		}
		else {

			/*			boundary cells			*/
			// initialize dispersion and metric block matrices
			MatrixXd bnd_dispBlock = MatrixXd::Zero(nNodes, 2 * nNodes); // boundary cell specific
			dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes);
			M_1M_rG = MatrixXd::Zero(nNodes, nNodes + 2);
			MatrixXd bnd_M_1M_rG = MatrixXd::Zero(nNodes, nNodes + 1); // boundary cell specific

			// auxiliary block [ d g(c) / d c ] for left boundary cell
			MatrixXd GBlock_l = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_l.block(0, 0, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_l(nNodes - 1, nNodes - 1) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock_l(nNodes - 1, nNodes) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock_l *= invMap;
			// auxiliary block [ d g(c) / d c ] for right boundary cell
			MatrixXd GBlock_r = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_r.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_r(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock_r(0, 1) += 0.5 * _disc.parInvWeights[parType][0];
			GBlock_r *= invMap;

			/*			 left boundary cell				*/
			int _cell = 0;
			// numerical flux contribution for right interface of left boundary cell -> d f^*_N / d cp
			MatrixXd bnd_gStarDC = MatrixXd::Zero(nNodes, 2 * nNodes);
			bnd_gStarDC.block(nNodes - 1, 0, 1, nNodes + 1) = GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 1) += GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;
			// dispBlock <- invMap * ( D * G_l - M^-1 * B * [G_l - g^*] )
			bnd_dispBlock.block(0, 0, nNodes, nNodes + 1) = (_disc.parPolyDerM[parType] * GBlock_l - _disc.parInvWeights[parType].asDiagonal() * B * GBlock_l);
			bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
			bnd_dispBlock *= invMap;

			// compute metric part separately, because of cell dependend left interface coordinate r_L
			double r_L = static_cast<double>(_parRadius[parType]) - _cell * _disc.deltaR[parType]; // left boundary of current cell

			for (int node = 0; node < nNodes; node++) {
				for (int node2 = 0; node2 < nNodes + 1; node2++) {
					bnd_M_1M_rG(node, node2) = GBlock_l(node, node2) * (2.0 / (r_L + (_disc.deltaR[parType] / 2.0) * (_disc.parNodes[parType][node] + 1.0)));
				}
			}

			// dispBlock <- M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] )
			bnd_dispBlock.block(0, 0, nNodes, nNodes + 1) += bnd_M_1M_rG;

			//std::cout << "dispBlock\n" << dispBlock * 1e-5 << std::endl;

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				// get bound and liquid diffusion coefficients for current component
				Eigen::VectorXd diff = getParDiffComp(parType, comp, parDiff, parSurfDiff);
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						jacP.coeffRef(comp * sComp + i * sNode,
									  comp * sComp + j * sNode)
							= -diff[0] * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								jacP.coeffRef(comp * sComp + i * sNode,
									idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
									= -diff[bnd + 1] * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
							}
						}
					}
				}
			}

			/*			 right boundary cell				*/
			_cell = _disc.nParCell[parType] - 1;
			// numerical flux contribution for left interface of right boundary cell -> d f^*_0 / d cp
			bnd_gStarDC.setZero();
			bnd_gStarDC.block(0, nNodes - 1, 1, nNodes + 1) = GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC.block(0, 0, 1, nNodes + 1) += GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;
			// dispBlock <- invMap * ( D * G_r - M^-1 * B * [G_r - g^*] )
			bnd_dispBlock.setZero();
			bnd_dispBlock.block(0, nNodes - 1, nNodes, nNodes + 1) = (_disc.parPolyDerM[parType] * GBlock_r - _disc.parInvWeights[parType].asDiagonal() * B * GBlock_r);
			bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
			bnd_dispBlock *= invMap;

			// compute metric part separately, because of cell dependend left interface coordinate r_L
			r_L = static_cast<double>(_parRadius[parType]) - _cell * _disc.deltaR[parType]; // left boundary of current cell
			bnd_M_1M_rG.setZero();
			for (int node = 0; node < nNodes; node++) {
				for (int node2 = 0; node2 < nNodes + 1; node2++) {
					bnd_M_1M_rG(node, node2) = GBlock_r(node, node2) * (2.0 / (r_L + (_disc.deltaR[parType] / 2.0) * (_disc.parNodes[parType][node] + 1.0)));
				}
			}

			// dispBlock <- M^-1 * M_r * G_r +  invMap * ( D * G_r - M^-1 * B * [G_r - g^*] )
			bnd_dispBlock.block(0, nNodes - 1, nNodes, nNodes + 1) += bnd_M_1M_rG;

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
						// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
						jacP.coeffRef(comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
									  comp * sComp + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
							= -parDiff[0] * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
								jacP.coeffRef(comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
									idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
									= -parSurfDiff[bnd] * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
							}
						}
					}
				}
			}

			/*				inner cells				*/

			// auxiliary block [ d g(c) / d c ] for inner cells
			MatrixXd GBlock = MatrixXd::Zero(nNodes, nNodes + 2);
			GBlock.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock(0, 1) += 0.5 * _disc.parInvWeights[parType][0];
			GBlock(nNodes - 1, nNodes) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock(nNodes - 1, nNodes + 1) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock *= invMap;

			// numerical flux contribution
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes);
			gStarDC.block(0, nNodes - 1, 1, nNodes + 2) = GBlock.block(0, 0, 1, nNodes + 2);
			gStarDC.block(0, 0, 1, nNodes + 1) += GBlock.block(nNodes - 1, 1, 1, nNodes + 1);
			gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += GBlock.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC.block(nNodes - 1, 2 * nNodes - 1, 1, nNodes + 1) += GBlock.block(0, 0, 1, nNodes + 1);
			gStarDC *= 0.5;

			dispBlock.setZero();
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = (_disc.parPolyDerM[parType] * GBlock - _disc.parInvWeights[parType].asDiagonal() * B * GBlock);
			dispBlock.block(0, 0, nNodes, 3 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * gStarDC;
			dispBlock *= invMap;

			for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {

				// compute metric part separately, because of cell dependend left interface coordinate r_L
				r_L = static_cast<double>(_parRadius[parType]) - cell * _disc.deltaR[parType]; // left boundary of current cell
				for (int node = 0; node < nNodes; node++) {
					for (int node2 = 0; node2 < nNodes + 2; node2++) {
						M_1M_rG(node, node2) = GBlock(node, node2) * (2.0 / (r_L + (_disc.deltaR[parType] / 2.0) * (_disc.parNodes[parType][node] + 1.0)));
					}
				}
				// dispBlock <- M^-1 * M_r * G_r +  invMap * ( D * G_r - M^-1 * B * [G_r - g^*] )
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) += M_1M_rG;

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// handle liquid state
							// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
							// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
							jacP.coeffRef(comp * sComp + cell * sCell + i * sNode,
										  comp * sComp + (cell - 1) * sCell + j * sNode)
								= -parDiff[0] * dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {
								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
									// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
									jacP.coeffRef(comp * sComp + cell * sCell + i * sNode,
										idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd
										+ (cell - 1) * sCell + j * sNode)
										= -parSurfDiff[bnd] * invBetaP[comp] * dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
								}
							}
						}
					}
				}
				// set back to: dispBlock <- invMap * ( D * G_l - M^-1 * B * [G_l - g^*] )
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) -= M_1M_rG;
			}

		} // if nCells > 1
		return 0;
	}

	/**
	* @brief calculates the number of non zeros for the DG convection dispersion jacobian
	* @detail only dispersion entries are relevant as the convection entries are a subset of these
	*/
	unsigned int calcConvDispNNZ(Discretization disc) {

		if (disc.modal) {
			return disc.nComp * ((3u * disc.nCol - 2u) * disc.nNodes * disc.nNodes + (2u * disc.nCol - 3u) * disc.nNodes);
		}
		else {
			return disc.nComp * (disc.nCol * disc.nNodes * disc.nNodes + 8u * disc.nNodes);
		}
	}
	unsigned int calcParDispNNZ(int parType, Discretization disc) {

		if (disc.parModal) {
			return disc.nComp * ((3u * disc.nParCell[parType] - 2u) * disc.nParNode[parType] * disc.nParNode[parType] + (2u * disc.nParCell[parType] - 3u) * disc.nParNode[parType]);
		}
		else {
			return disc.nComp * (disc.nParCell[parType] * disc.nParNode[parType] * disc.nParNode[parType] + 8u * disc.nParNode[parType]);
		}
	}

	/**
		* @brief analytically calculates the convection dispersion jacobian for the nodal DG scheme
		*/
	int calcConvDispNodalJacobian(Eigen::SparseMatrix<double, RowMajor>& jac) {

		Indexer idx(_disc);

		int sNode = idx.strideColNode();
		int sCell = idx.strideColCell();
		int sComp = idx.strideColComp();
		int offC = 0; // inlet DOFs not included in Jacobian

		unsigned int nNodes = _disc.nNodes;
		unsigned int polyDeg = _disc.polyDeg;
		unsigned int nCells = _disc.nCol;
		unsigned int nComp = _disc.nComp;

		/*======================================================*/
		/*			Compute Dispersion Jacobian Block			*/
		/*======================================================*/

		/*		Inner cell dispersion blocks		*/

		// auxiliary Block for [ d g(c) / d c ], needed in Dispersion block
		MatrixXd GBlock = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlock.block(0, 1, nNodes, nNodes) = _disc.polyDerM;
		GBlock(0, 0) -= 0.5 * _disc.invWeights[0];
		GBlock(0, 1) += 0.5 * _disc.invWeights[0];
		GBlock(nNodes - 1, nNodes) -= 0.5 * _disc.invWeights[polyDeg];
		GBlock(nNodes - 1, nNodes + 1) += 0.5 * _disc.invWeights[polyDeg];
		GBlock *= 2 / _disc.deltaZ;

		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes); //
		// NOTE: N = polyDeg
		//cell indices : 0	 , ..., nNodes - 1;	nNodes, ..., 2 * nNodes - 1;	2 * nNodes, ..., 3 * nNodes - 1
		//			j  : -N-1, ..., -1		  ; 0     , ..., N			   ;	N + 1, ..., 2N + 1
		dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = _disc.polyDerM * GBlock;
		dispBlock(0, nNodes - 1) += -_disc.invWeights[0] * (-0.5 * GBlock(0, 0) + 0.5 * GBlock(nNodes - 1, nNodes)); // G_N,N		i=0, j=-1
		dispBlock(0, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlock(0, 1) + 0.5 * GBlock(nNodes - 1, nNodes + 1)); // G_N,N+1	i=0, j=0
		dispBlock.block(0, nNodes + 1, 1, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlock.block(0, 2, 1, nNodes)); // G_i,j		i=0, j=1,...,N+1
		dispBlock.block(0, 0, 1, nNodes - 1) += -_disc.invWeights[0] * (0.5 * GBlock.block(nNodes - 1, 1, 1, nNodes - 1)); // G_N,j+N+1		i=0, j=-N-1,...,-2
		dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlock.block(nNodes - 1, 0, 1, nNodes)); // G_i,j+N+1		i=N, j=--1,...,N-1
		dispBlock(nNodes - 1, 2 * nNodes - 1) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlock(nNodes - 1, nNodes) + 0.5 * GBlock(0, 0)); // G_i,j		i=N, j=N
		dispBlock(nNodes - 1, 2 * nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlock(nNodes - 1, nNodes + 1) + 0.5 * GBlock(0, 1)); // G_i,j		i=N, j=N+1
		dispBlock.block(nNodes - 1, 2 * nNodes + 1, 1, nNodes - 1) += _disc.invWeights[nNodes - 1] * (0.5 * GBlock.block(0, 2, 1, nNodes - 1)); // G_0,j-N-1		i=N, j=N+2,...,2N+1
		dispBlock *= 2 / _disc.deltaZ;

		// insert Blocks to Jacobian inner cells (only for nCells >= 3)
		if (nCells >= 3u) {
			for (unsigned int cell = 1; cell < nCells - 1; cell++) {
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry
							_jacC.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
								offC + (cell - 1) * sCell + comp * sComp + j * sNode)
								= -dispBlock(i, j) * _disc.dispersion[comp];
						}
					}
				}
			}
		}

		/*				Boundary cell Dispersion blocks			*/

		/* left cell */
		// adjust auxiliary Block [ d g(c) / d c ] for left boundary cell
		MatrixXd GBlockBound = GBlock;
		GBlockBound(0, 1) -= 0.5 * _disc.invWeights[0] * 2 / _disc.deltaZ;

		// estimate dispersion block ( j < 0 not needed)
		dispBlock.setZero();
		dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = _disc.polyDerM * GBlockBound;
		dispBlock.block(0, nNodes - 1, 1, nNodes + 2) += -_disc.invWeights[0] * (-GBlockBound.block(0, 0, 1, nNodes + 2)); // G_N,N		i=0, j=-1,...,N+1
		dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound.block(nNodes - 1, 0, 1, nNodes)); // G_i,j+N+1		i=N, j=--1,...,N-1
		dispBlock(nNodes - 1, 2 * nNodes - 1) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound(nNodes - 1, nNodes) + 0.5 * GBlockBound(0, 0)); // G_i,j		i=N, j=N
		dispBlock(nNodes - 1, 2 * nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound(nNodes - 1, nNodes + 1) + 0.5 * GBlock(0, 1)); // G_i,j		i=N, j=N+1
		dispBlock.block(nNodes - 1, 2 * nNodes + 1, 1, nNodes - 1) += _disc.invWeights[nNodes - 1] * (0.5 * GBlock.block(0, 2, 1, nNodes - 1)); // G_0,j-N-1		i=N, j=N+2,...,2N+1
		dispBlock *= 2 / _disc.deltaZ;
		if (nCells != 1u) { // "standard" case
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes; j < dispBlock.cols(); j++) {
						_jacC.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - nNodes) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}
		else { // special case
			dispBlock.setZero();
			dispBlock.block(0, nNodes, nNodes, nNodes) = _disc.polyDerM * _disc.polyDerM;
			dispBlock *= 2 / _disc.deltaZ;
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes; j < nNodes * 2u; j++) {
						_jacC.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - nNodes) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}

		/* right cell */
		if (nCells != 1u) { // "standard" case
	   // adjust auxiliary Block [ d g(c) / d c ] for left boundary cell
			GBlockBound(0, 1) += 0.5 * _disc.invWeights[0] * 2 / _disc.deltaZ; 	// reverse change from left boundary
			GBlockBound(nNodes - 1, nNodes) += 0.5 * _disc.invWeights[polyDeg] * 2 / _disc.deltaZ;

			// estimate dispersion block (only estimation differences to inner cell at N = 0 and j > N not needed)
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = _disc.polyDerM * GBlockBound;
			dispBlock(0, nNodes - 1) += -_disc.invWeights[0] * (-0.5 * GBlockBound(0, 0) + 0.5 * GBlock(nNodes - 1, nNodes)); // G_N,N		i=0, j=-1
			dispBlock(0, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlockBound(0, 1) + 0.5 * GBlock(nNodes - 1, nNodes + 1)); // G_N,N+1	i=0, j=0
			dispBlock.block(0, nNodes + 1, 1, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlockBound.block(0, 2, 1, nNodes)); // G_i,j		i=0, j=1,...,N+1
			dispBlock.block(0, 0, 1, nNodes - 1) += -_disc.invWeights[0] * (0.5 * GBlock.block(nNodes - 1, 1, 1, nNodes - 1)); // G_N,j+N+1		i=0, j=-N-1,...,-2
			dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += _disc.invWeights[nNodes - 1] * (-GBlockBound.block(nNodes - 1, 0, 1, nNodes + 2)); // G_i,j+N+1		i=N, j=--1,...,N+1
			dispBlock *= 2 / _disc.deltaZ;
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						_jacC.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
									   offC + (nCells - 1 - 1) * sCell + comp * sComp + j * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		} // "standard" case
		/*======================================================*/
		/*			Compute Convection Jacobian Block			*/
		/*======================================================*/

		// Convection block [ d RHS_conv / d c ], also depends on first entry of previous cell
		MatrixXd convBlock = MatrixXd::Zero(nNodes, nNodes + 1);
		convBlock.block(0, 1, nNodes, nNodes) -= _disc.polyDerM;
		convBlock(0, 0) += _disc.invWeights[0];
		convBlock(0, 1) -= _disc.invWeights[0];
		convBlock *= 2 * _disc.velocity / _disc.deltaZ;

		// special inlet DOF treatment for first cell
		_jacInlet(0, 0) = -convBlock(0, 0); // only first node depends on inlet concentration
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < convBlock.rows(); i++) {
				//_jac.coeffRef(offC + comp * sComp + i * sNode, comp * sComp) = -convBlock(i, 0); // dependency on inlet DOFs is handled in _jacInlet
				for (unsigned int j = 1; j < convBlock.cols(); j++) {
					_jacC.coeffRef(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - 1) * sNode)
						+= -convBlock(i, j);
				}
			}
		}
		for (unsigned int cell = 1; cell < nCells; cell++) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < convBlock.rows(); i++) {
					//Eigen::SparseMatrix<double, RowMajor>::InnerIterator it(_jac, offC + cell * sCell + comp * sComp + i * sNode);
					//it += _disc.polyDeg; // jump over dispersion block entries
					for (unsigned int j = 0; j < convBlock.cols(); j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
						// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
						_jacC.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
							offC + cell * sCell - sNode + comp * sComp + j * sNode)
							+= -convBlock(i, j);
						//it.valueRef() += -convBlock(i, j);
						//++it;
					}
				}
			}
		}

		return 0;
	}

	Eigen::MatrixXd getGBlock() {

		int nNodes = _disc.nNodes;
		// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
		MatrixXd gBlock = MatrixXd::Zero(nNodes, nNodes + 2);
		gBlock.block(0, 1, nNodes, nNodes) = _disc.polyDerM;
		gBlock.block(0, 0, nNodes, 1) -= 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		gBlock.block(0, 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		gBlock.block(0, nNodes, nNodes, 1) -= 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		gBlock.block(0, nNodes + 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		gBlock *= 2 / _disc.deltaZ;

		return gBlock;
	}

	Eigen::MatrixXd getBMatrix() {

		MatrixXd B = MatrixXd::Zero(_disc.nNodes, _disc.nNodes);
		B(0, 0) = -1.0;
		B(_disc.nNodes - 1, _disc.nNodes - 1) = 1.0;

		return B;
	}

	Eigen::MatrixXd innerCellBlock() {

		int nNodes = _disc.nNodes;
		// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
		MatrixXd gBlock = getGBlock();

		// B matrix from DG scheme
		MatrixXd B = getBMatrix();

		// Inner dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		// NOTE: N = polyDeg
		//  indices  gStarDC    :     0   ,   1   , ..., nNodes; nNodes+1, ..., 2 * nNodes;	2*nNodes+1, ..., 3 * nNodes; 3*nNodes+1
		//	derivative index j  : -(N+1)-1, -(N+1),... ,  -1   ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
		// auxiliary block [d g^* / d c]
		gStarDC.block(0, nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC.block(0, 0, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC *= 0.5;

		//  indices  dispBlock :   0	 ,   1   , ..., nNodes;	nNodes+1, ..., 2 * nNodes;	2*nNodes+1, ..., 3 * nNodes; 3*nNodes+1
		//	derivative index j  : -(N+1)-1, -(N+1),...,	 -1	  ;   0     , ...,		N	 ;	  N + 1	  , ..., 2N + 2    ; 2(N+1) +1
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * gBlock - _disc.invMM * B * gBlock;
		dispBlock += _disc.invMM * B * gStarDC;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd leftBndryCellNghbrBlock() {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_l = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_l.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_l.block(0, nNodes, nNodes, 1) -= 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l.block(0, nNodes + 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		gStarDC.block(0, nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC.block(0, 0, 1, nNodes + 2) += GBlockBound_l.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC *= 0.5;
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * gBlock - _disc.invMM * B * gBlock;
		dispBlock += _disc.invMM * B * gStarDC;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd rightBndryCellNghbrBlock() {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_r = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_r.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_r.block(0, 0, nNodes, 1) -= 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r.block(0, 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		gStarDC.block(0, nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC.block(0, 0, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += GBlockBound_r.block(0, 0, 1, nNodes + 2);
		gStarDC *= 0.5;
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * gBlock - _disc.invMM * B * gBlock;
		dispBlock += _disc.invMM * B * gStarDC;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd leftBndryCellBlock() {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_l = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_l.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_l.block(0, nNodes, nNodes, 1) -= 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l.block(0, nNodes + 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += GBlockBound_l.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC *= 0.5;
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * GBlockBound_l - _disc.invMM * B * GBlockBound_l;
		dispBlock.block(0, nNodes + 1, nNodes, 2 * nNodes + 1) += _disc.invMM * B * gStarDC.block(0, nNodes + 1, nNodes, 2 * nNodes + 1);
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd rightBndryCellBlock() {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_r = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_r.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_r.block(0, 0, nNodes, 1) -= 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r.block(0, 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		gStarDC.block(0, nNodes, 1, nNodes + 2) += GBlockBound_r.block(0, 0, 1, nNodes + 2);
		gStarDC.block(0, 0, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC *= 0.5;
		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * GBlockBound_r - _disc.invMM * B * GBlockBound_r;
		dispBlock += _disc.invMM * B * gStarDC;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd getConvBlock() {

		int nNodes = _disc.nNodes;
		// Convection block [ d RHS_conv / d c ], additionally depends on first entry of previous cell
		MatrixXd convBlock = MatrixXd::Zero(nNodes, nNodes + 1);
		convBlock.block(0, 0, nNodes, 1) += _disc.invMM.block(0, 0, nNodes, 1);
		convBlock.block(0, 1, nNodes, nNodes) -= _disc.polyDerM;
		convBlock.block(0, 1, nNodes, 1) -= _disc.invMM.block(0, 0, nNodes, 1);
		convBlock *= 2 * _disc.velocity / _disc.deltaZ;

		return convBlock;
	}

	Eigen::MatrixXd specialBlockOneCell() {

		int nNodes = _disc.nNodes;
		// Auxiliary Block [ d g(c) / d c ], additionally depends on boundary entries of neighbouring cells
		MatrixXd gBlock = MatrixXd::Zero(nNodes, nNodes + 2);
		gBlock.block(0, 1, nNodes, nNodes) = _disc.polyDerM;
		gBlock *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ] equals zero.
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * gBlock - _disc.invMM * B * gBlock;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	Eigen::MatrixXd specialBlockTwoCells(bool right) {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_l = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_l.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_l.block(0, nNodes, nNodes, 1) -= 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l.block(0, nNodes + 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l *= 2 / _disc.deltaZ;
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_r = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_r.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_r.block(0, 0, nNodes, 1) -= 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r.block(0, 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r *= 2 / _disc.deltaZ;

		if (right) { // right boundary cell

			// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
			gStarDC.block(0, nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
			gStarDC.block(0, 0, 1, nNodes + 2) += GBlockBound_l.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC *= 0.5;
			// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
			dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * GBlockBound_r - _disc.invMM * B * GBlockBound_r;
			dispBlock += _disc.invMM * B * gStarDC;
			dispBlock *= 2 / _disc.deltaZ;

			return dispBlock;
		}
		else { // left boundary cell

			// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
			gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += GBlockBound_r.block(0, 0, 1, nNodes + 2);
			gStarDC *= 0.5;
			// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
			MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
			dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * GBlockBound_l - _disc.invMM * B * GBlockBound_l;
			dispBlock += _disc.invMM * B * gStarDC;
			dispBlock *= 2 / _disc.deltaZ;

			return dispBlock;
		}
	}

	Eigen::MatrixXd specialBlockThreeCells() {

		int nNodes = _disc.nNodes;
		MatrixXd gBlock = getGBlock();
		// B matrix from DG scheme
		MatrixXd B = getBMatrix();
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_l = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_l.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_l.block(0, nNodes, nNodes, 1) -= 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l.block(0, nNodes + 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, nNodes - 1, nNodes, 1);
		GBlockBound_l *= 2 / _disc.deltaZ;
		// boundary auxiliary block [ d g(c) / d c ]
		MatrixXd GBlockBound_r = MatrixXd::Zero(nNodes, nNodes + 2);
		GBlockBound_r.block(0, 1, nNodes, nNodes) += _disc.polyDerM;
		GBlockBound_r.block(0, 0, nNodes, 1) -= 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r.block(0, 1, nNodes, 1) += 0.5 * _disc.invMM.block(0, 0, nNodes, 1);
		GBlockBound_r *= 2 / _disc.deltaZ;
		// auxiliary block [ d g^* / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes + 2);
		gStarDC.block(0, nNodes, 1, nNodes + 2) += gBlock.block(0, 0, 1, nNodes + 2);
		gStarDC.block(0, 0, 1, nNodes + 2) += GBlockBound_l.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, nNodes, 1, nNodes + 2) += gBlock.block(nNodes - 1, 0, 1, nNodes + 2);
		gStarDC.block(nNodes - 1, 2 * nNodes, 1, nNodes + 2) += GBlockBound_r.block(0, 0, 1, nNodes + 2);
		gStarDC *= 0.5;

		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //
		dispBlock.block(0, nNodes, nNodes, nNodes + 2) += _disc.polyDerM * gBlock - _disc.invMM * B * gBlock;
		dispBlock += _disc.invMM * B * gStarDC;
		dispBlock *= 2 / _disc.deltaZ;

		return dispBlock;
	}

	/**
		* @brief analytically calculates the convection dispersion jacobian for the modal DG scheme
		*/
	int calcConvDispModalJacobian(Eigen::SparseMatrix<double, RowMajor>& jac) {

		Indexer idx(_disc);

		int sNode = idx.strideColNode();
		int sCell = idx.strideColCell();
		int sComp = idx.strideColComp();
		int offC = 0; // inlet DOFs not included in Jacobian

		unsigned int nNodes = _disc.nNodes;
		unsigned int nCells = _disc.nCol;
		unsigned int nComp = _disc.nComp;

		/*======================================================*/
		/*			Compute Dispersion Jacobian Block			*/
		/*======================================================*/

		// Dispersion block [ d RHS_disp / d c ], depends on whole previous and subsequent cell plus first entries of subsubsequent cells
		MatrixXd dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes + 2); //

		/* Inner cells (exist only if nCells >= 5) */
		if (nCells >= 5) {

			dispBlock = innerCellBlock();

			for (unsigned int cell = 2; cell < nCells - 2; cell++) {
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry
							_jacC.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
								offC + cell * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode)
								= -dispBlock(i, j) * _disc.dispersion[comp];
						}
					}
				}
			}

		}

		/*	boundary cell neighbours (exist only if nCells >= 4)	*/
		if (nCells >= 4) {
			// left boundary cell neighbour

			dispBlock = leftBndryCellNghbrBlock();

			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 1; j < dispBlock.cols(); j++) {
						// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry. Also adjust for iterator j (-1)
						_jacC.coeffRef(offC + nNodes * sNode + comp * sComp + i * sNode,
							offC + comp * sComp + (j - 1) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}

			// right boundary cell neighbour

			dispBlock = rightBndryCellNghbrBlock();

			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < dispBlock.cols() - 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + (nCells - 2) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 2) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}

		}

		/*			boundary cells (exist only if nCells >= 3)			*/
		if (nCells >= 3) {
			// left boundary cell

			dispBlock = leftBndryCellBlock();
			unsigned int special = 0u; if (nCells == 3u) special = 1u; // limits the iterator for special case nCells = 3
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes + 1; j < dispBlock.cols() - special; j++) {
						// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - (nNodes + 1)) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}

			// right boundary cell

			dispBlock = rightBndryCellBlock();

			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = special; j < 2 * nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell and one node, add component offset and go node strides from there for relevant dispersion block entries.
						_jacC.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 1) * sCell - (nNodes + 1) * sNode + comp * sComp + j * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}

		/* For special cases nCells = 1, 2, 3, some cells still have to be treated separately*/

		if (nCells == 1) {
			dispBlock = specialBlockOneCell();
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes + 1; j < 2 * nNodes + 1; j++) {
						// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - (nNodes + 1)) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}
		else if (nCells == 2) {
			dispBlock = specialBlockTwoCells(0); // get left boundary cell
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes + 1; j < 3 * nNodes + 1; j++) {
						// row: jump over inlet DOFs, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs, add component offset, adjust for iterator j (-Nnodes-1) and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - (nNodes + 1)) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
			dispBlock = specialBlockTwoCells(1); // get right boundary cell
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 1; j < 2 * nNodes + 1; j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cells, go back one cell, add component offset, adjust for iterator (j-1) and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
							offC + (nCells - 1) * sCell - (nNodes)*sNode + comp * sComp + (j - 1) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}
		else if (nCells == 3) {
			dispBlock = specialBlockThreeCells();
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 1; j < dispBlock.cols() - 1; j++) {
						// row: jump over inlet DOFs and previous cell, add component offset and go node strides from there for each dispersion block entry
						// col: jump over inlet DOFs and previous cell, go back one cell, add component offset, adjust for iterator (j-1) and go node strides from there for each dispersion block entry.
						_jacC.coeffRef(offC + 1 * sCell + comp * sComp + i * sNode,
							offC + 1 * sCell - (nNodes)*sNode + comp * sComp + (j - 1) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}

		/*======================================================*/
		/*			Compute Convection Jacobian Block			*/
		/*======================================================*/

		// Convection block [ d RHS_conv / d c ], additionally depends on first entry of previous cell
		MatrixXd convBlock = MatrixXd::Zero(nNodes, nNodes + 1);
		convBlock = getConvBlock();

		// special inlet DOF treatment for first cell
		_jacInlet = -convBlock.col(0); // only first cell depends on inlet concentration
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < convBlock.rows(); i++) {
				//_jac.coeffRef(offC + comp * sComp + i * sNode, comp * sComp) = -convBlock(i, 0); // dependency on inlet DOFs is handled in _jacInlet
				for (unsigned int j = 1; j < convBlock.cols(); j++) {
					_jacC.coeffRef(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - 1) * sNode)
						-= convBlock(i, j);
				}
			}
		}
		for (unsigned int cell = 1; cell < nCells; cell++) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < convBlock.rows(); i++) {
					for (unsigned int j = 0; j < convBlock.cols(); j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
						// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
						_jacC.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
							offC + cell * sCell - sNode + comp * sComp + j * sNode)
							-= convBlock(i, j);
					}
				}
			}
		}

		return 0;
	}

	/**
	* @brief adds time derivative to the bulk jacobian
	* @detail alpha * d Bulk_Residual / d c_t = alpha * I is added to the bulk jacobian
	*/
	void addTimeDerBulkJacobian(double alpha, Indexer idxr) {

		unsigned int offC = 0; // inlet DOFs not included in Jacobian

		for (linalg::BandedEigenSparseRowIterator jac(_jacCdisc, offC); jac.row() < _disc.nComp * _disc.nPoints; ++jac) {

			jac[0] += alpha; // main diagonal

		}
	}

};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_GENERALRATEMODELDG_HPP_
