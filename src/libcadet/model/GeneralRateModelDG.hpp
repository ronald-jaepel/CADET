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
//TODO: delete iostream, iomanip
#include <iostream>
#include <iomanip>      // std::setprecision
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
	//void assembleOffdiagJacFluxParticle(double t, unsigned int secIdx, double const* vecStateY); // TODO delete
	void extractJacobianFromAD(active const* const adRes, unsigned int adDirOffset);

	void assembleDiscretizedGlobalJacobian(double alpha, Indexer idxr);

	void setEquidistantRadialDisc(unsigned int parType);
	//@TODO? allow different cell spacing for DG
	void setEquivolumeRadialDisc(unsigned int parType);
	void setUserdefinedRadialDisc(unsigned int parType);
	void updateRadialDisc();

	void addTimeDerivativeToJacobianParticleShell(linalg::BandedEigenSparseRowIterator& jac, const Indexer& idxr, double alpha, unsigned int parType);
	
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
		bool exactInt;	//!< 1 for exact integration, 0 for inexact LGL quadrature
		unsigned int nParType; //!< Number of particle types
		unsigned int* nParCell; //!< Array with number of radial cells in each particle type
		unsigned int* nParPointsBeforeType; //!< Array with total number of radial points before a particle type (cumulative sum of nParPoints), additional last element contains total number of particle shells
		unsigned int* parPolyDeg; //!< polynomial degree of particle elements
		unsigned int* nParNode; //!< Array with number of radial nodes per cell in each particle type
		unsigned int* nParPoints; //!< Array with number of radial nodes per cell in each particle type
		bool* parExactInt;	//!< 1  for exact integration, 0 for inexact LGL quadrature for each particle type
		unsigned int* parTypeOffset; //!< Array with offsets (in particle block) to particle type, additional last element contains total number of particle DOFs
		unsigned int* nBound; //!< Array with number of bound states for each component and particle type (particle type major ordering)
		unsigned int* boundOffset; //!< Array with offset to the first bound state of each component in the solid phase (particle type major ordering)
		unsigned int* strideBound; //!< Total number of bound states for each particle type, additional last element contains total number of bound states for all types
		unsigned int* nBoundBeforeType; //!< Array with number of bound states before a particle type (cumulative sum of strideBound)

		const double SurfVolRatioSlab = 1.0; //!< Surface to volume ratio for a slab-shaped particle
		const double SurfVolRatioSphere = 3.0; //!< Surface to volume ratio for a spherical particle

		//////////////////////		DG specifics		////////////////////////////////////////////////////#
		//
		//// NOTE: no different Riemann solvers or boundary conditions

		double deltaZ; //!< equidistant column spacing
		double* deltaR; //!< equidistant particle radial spacing for each particle type
		Eigen::VectorXd nodes; //!< Array with positions of nodes in reference element
		Eigen::MatrixXd polyDerM; //!< Array with polynomial derivative Matrix
		Eigen::VectorXd invWeights; //!< Array with weights for numerical quadrature of size nNodes
		Eigen::MatrixXd invMM; //!< dense inverse mass matrix for exact integration
		Eigen::VectorXd* parNodes; //!< Array with positions of nodes in radial reference element for each particle
		Eigen::MatrixXd* parPolyDerM; //!< Array with polynomial derivative Matrix for each particle
		Eigen::VectorXd* parInvWeights; //!< Array with weights for LGL quadrature of size nNodes for each particle
		Eigen::MatrixXd* parInvMM; //!< dense !INVERSE! mass matrix for exact integration for each particle
		Eigen::VectorXd* Ir; //!< metric part for each particle type and cell, particle type major ordering
		Eigen::MatrixXd* Dr; //!< derivative matrices including metrics for each particle type and cell, particle type major ordering
		Eigen::VectorXi offsetMetric; //!< offset in metrics Ir, Dr -> summed up nCells of all previous parTypes

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
		Eigen::VectorXd h; //!< auxiliary variable h = vc - D_ax g
		Eigen::VectorXd surfaceFlux; //!< stores the surface flux values of the bulk phase
		Eigen::VectorXd* surfaceFluxParticle; //!< stores the surface flux values for each particle
		Eigen::Vector4d boundary; //!< stores the boundary values from Danckwert boundary conditions of the bulk phase
		const double* localFlux; //!< stores the local Flux to implement film diffusion

		std::vector<bool> isKinetic;

		bool newStaticJac; //!< determines wether static analytical jacobian needs to be computed (every section)

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
			localFlux = new double[1];

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
			}

			// prepare metric part for particles. To be computed in updateRadialDisc()
			offsetMetric = VectorXi::Zero(nParType); // auxiliary counter
			for (int parType = 1; parType < nParType; parType++) {
				offsetMetric[parType] += nParCell[parType - 1];
			}
			Dr = new MatrixXd[offsetMetric.sum() + nParCell[0]];
			Ir = new VectorXd[offsetMetric.sum() + nParCell[0]];

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
			for (unsigned int j = 1; j <= _polyDeg; j++) {
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

	Eigen::SparseLU<Eigen::SparseMatrix<double>> _globalSolver; //!< linear solver
	//Eigen::BiCGSTAB<Eigen::SparseMatrix<double, RowMajor>, Eigen::DiagonalPreconditioner<double>> _globalSolver;

	Eigen::SparseMatrix<double, RowMajor> _globalJac; //!< static part of global Jacobian
	Eigen::SparseMatrix<double, RowMajor> _globalJacDisc; //!< global Jacobian with time derivative from BDF method

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

	ArrayPool _discParFlux; //!< Storage for discretized @f$ k_f @f$ value //TODO delete

	bool _factorizeJacobian; //!< Determines whether the Jacobian needs to be factorized
	double* _tempState; //!< Temporary storage with the size of the state vector or larger if binding models require it
	linalg::Gmres _gmres; //!< GMRES algorithm for the Schur-complement in linearSolve() //TODO delete
	double _schurSafety; //!< Safety factor for Schur-complement solution //TODO delete
	int _colParBoundaryOrder; //!< Order of the bulk-particle boundary discretization //TODO delete

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

	// Wrapper for calling the corresponding function in GeneralRateModelDG class //TODO: delete!
	//friend int schurComplementMultiplierGRM_DG(void* userData, double const* x, double* z);

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
		inline int strideParBlock(int parType) const CADET_NOEXCEPT { return static_cast<int>(_disc.nParPoints[parType]) * strideParShell(parType); }

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
		inline int offsetJf(ParticleTypeIndex pti, ParticleIndex pi) const CADET_NOEXCEPT { return offsetJf(pti) + pi.value * _disc.nComp; }
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
			// Note that the DG particle nodes are oppositely ordered compared to the FV particle cells
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

	void parVolumeIntegral(const int parType, const bool aux, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state, Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer) {

		int nNodes = _disc.nParNode[parType];

		/* no additional metric term for auxiliary equation -> res = - D * (d_p * c^p + invBeta_p sum_mi d_s c^s) */
		if (aux) {
			// comp-cell-node state vector: use of Eigen lib performance
			for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
				stateDer.segment(Cell * nNodes, nNodes)
					-= _disc.parPolyDerM[parType] * state.segment(Cell * nNodes, nNodes);
			}
		}
		/* include metrics for main particle equation -> res = - D * (d_p * c^p + invBeta_p sum_mi d_s c^s) */
		else {

			int Cell0 = 0; // auxiliary variable to distinguish special case

			// special case for non slab-shaped particles without core => r(xi_0) = 0
			if (_parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0) {
				Cell0 = 1;

				// estimate volume integral except for boundary node
				stateDer.segment(1, nNodes - 1) -= _disc.Dr[_disc.offsetMetric[parType]].block(1, 1, nNodes - 1, nNodes - 1) * state.segment(1, nNodes - 1);
				// estimate volume integral for boundary node: sum_{j=1}^N state_j * w_j * D_{j,0} * r_j
				stateDer[0] += (state.segment(1, nNodes - 1).array()
					* _disc.parInvWeights[parType].segment(1, nNodes - 1).array().cwiseInverse()
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1).array()
					* _disc.Ir[_disc.offsetMetric[parType]].segment(1, nNodes - 1).array()
					).sum();
			}

			// "standard" computation for remaining cells
			for (int cell = Cell0; cell < _disc.nParCell[parType]; cell++) {
				stateDer.segment(cell * nNodes, nNodes) -= _disc.Dr[_disc.offsetMetric[parType] + cell] * state.segment(cell * nNodes, nNodes);
			}
		}
	}

	/*
	* @brief calculates the interface fluxes h* of bulk mass balance equation
	*/
	void InterfaceFlux(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, const VectorXd& g, const unsigned int comp) {

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

	/*
	 * @brief calculates the interface fluxes g* of particle mass balance equation
	*/
	void InterfaceFluxParticle(int parType, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		const unsigned int strideCell, const unsigned int strideNode, const bool aux, const int comp) {

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
		if (aux) { // ghost nodes given by state^- := state^+ for auxiliary equation
			_disc.surfaceFluxParticle[parType][0] = state[0];

			_disc.surfaceFluxParticle[parType][_disc.nParCell[parType]] = state[_disc.nParCell[parType] * strideCell - strideNode];
		}
		else {
			
			// film diffusion BC
			_disc.surfaceFluxParticle[parType][_disc.nParCell[parType]] = _disc.localFlux[comp]
					/ (static_cast<double>(_parPorosity[parType]) * static_cast<double>(_poreAccessFactor[parType * _disc.nComp + comp]))
				* (2.0 / _disc.deltaR[parType]); // inverse squared mapping is also applied, so we apply Map * invMap^2 = invMap

			// inner particle BC
			_disc.surfaceFluxParticle[parType][0] = 0.0;

		}
	}

	/**
	* @brief calculates and fills the surface flux values for auxiliary equation
	* @param [in] strideCell component-wise cell stride
	* @param [in] strideNodecomponent-wise node stride
	*/
	void InterfaceFluxAuxiliary(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, const unsigned int strideCell, const unsigned int strideNode) {

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
	* @brief calculates the surface Integral, depending on the approach (exact/inexact integration)
	* @param [in] state relevant state vector
	* @param [in] stateDer state derivative vector the solution is added to
	* @param [in] aux true for auxiliary equation, false for main equation
		surfaceIntegral(cPtr, &(disc.g[0]), disc,&(disc.h[0]), resPtrC, 0, secIdx);
	* @param [in] strideCell component-wise cell stride
	* @param [in] strideNodecomponent-wise node stride
	*/
	void surfaceIntegral(Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& C, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer, const bool aux, const unsigned int Comp, const unsigned int strideCell, const unsigned int strideNode) {

		// calc numerical flux values c* or h* depending on equation switch aux
		(aux == 1) ? InterfaceFluxAuxiliary(C, strideCell, strideNode) : InterfaceFlux(C, _disc.g, Comp);
		if (_disc.exactInt) { // modal approach -> dense mass matrix
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

	/**
	 * @brief calculates the particle surface Integral (type- and component-wise)
	 * @param [in] parType current particle type
	 * @param [in] state relevant state vector
	 * @param [in] stateDer state derivative vector the solution is added to
	 * @param [in] aux true for auxiliary equation, false for main equation
		surfaceIntegral(cPtr, &(disc.g[0]), disc,&(disc.h[0]), resPtrC, 0, secIdx);
	 * @param [in] strideCell component-wise cell stride
	 * @param [in] strideNodecomponent-wise node stride
	 * @param [in] comp current component
	*/
	void parSurfaceIntegral(int parType, Eigen::Map<const VectorXd, 0, InnerStride<Dynamic>>& state,
		Eigen::Map<VectorXd, 0, InnerStride<Dynamic>>& stateDer, unsigned const int strideCell, unsigned const int strideNode, const bool aux, const int comp = 0) {

		// calc numerical flux values
		InterfaceFluxParticle(parType, state, strideCell, strideNode, aux, comp);

		// strong surface integral -> M^-1 B [state - state*]

		int Cell0 = 0; // auxiliary variable to distinguish special case
		// special case for sphere and cylinder if particle core = 0.0 -> leave out inner particle boundary flux
		if (_parGeomSurfToVol[parType] != _disc.SurfVolRatioSlab && _parCoreRadius[parType] == 0.0) {

			Cell0 = 1;

			stateDer[_disc.parPolyDeg[parType] * strideNode] // last cell node
				+= _disc.parInvWeights[parType][_disc.parPolyDeg[parType]] * (state[_disc.parPolyDeg[parType] * strideNode]
					- _disc.surfaceFluxParticle[parType][1]);
		}

		for (unsigned int Cell = Cell0; Cell < _disc.nParCell[parType]; Cell++) {

			stateDer[Cell * strideCell] // first cell node
				-= _disc.parInvWeights[parType][0] * (state[Cell * strideCell]
					- _disc.surfaceFluxParticle[parType][Cell]);

			stateDer[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode] // last cell node
				+= _disc.parInvWeights[parType][_disc.parPolyDeg[parType]] * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode]
					- _disc.surfaceFluxParticle[parType][Cell + 1u]);
		}
		// @TODO ? exact integration approach
		// if (!_disc.parExactInt[parType]) { // inexact integration approach -> diagonal mass matrix
		// ...
		// }
		//else { // exact integration approach -> dense mass matrix
		//	for (unsigned int Cell = 0; Cell < _disc.nParCell[parType]; Cell++) {
		//		// strong surface integral -> M^-1 B [state - state*]
		//		for (unsigned int Node = 0; Node < _disc.nParNode[parType]; Node++) {
		//			stateDer[Cell * strideCell + Node * strideNode]
		//				-= _disc.parInvMM[parType](Node, 0) * (state[Cell * strideCell]
		//					- _disc.surfaceFluxParticle[parType][Cell])
		//				- _disc.parInvMM[parType](Node, _disc.parPolyDeg[parType]) * (state[Cell * strideCell + _disc.parPolyDeg[parType] * strideNode]
		//					- _disc.surfaceFluxParticle[parType][(Cell + 1u)]);
		//		}
		//	}
		//}
	}

	/**
	* @brief calculates the substitute h = vc - sqrt(D_ax) g(c)
	*/
	void calcH(Eigen::Map<const VectorXd, 0, InnerStride<>>& C, const unsigned int Comp) {
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
	void applyMapping_Aux(Eigen::Map<VectorXd, 0, InnerStride<>>& state, const unsigned int Comp) {
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
		_disc.boundary[2] = -_disc.g[0]; // g_l inlet
		_disc.boundary[3] = -_disc.g[_disc.nPoints - 1]; // g_r outlet
	}

	/**
	 * @brief solves the auxiliary system g = d c / d xi
	 * @detail computes g = Dc - M^-1 B [c - c^*] and stores this in _disc.g_p
	*/
	void solve_auxiliary_DG(int parType, Eigen::Map<const VectorXd, 0, InnerStride<>>& conc, unsigned int strideCell, unsigned int strideNode, int comp) {

		Eigen::Map<VectorXd, 0, InnerStride<>> g_p(&_disc.g_p[parType][0], _disc.nParPoints[parType], InnerStride<>(1));

		// =================================================================//
		// solve auxiliary systems g = d c / d xi					        //
		// =================================================================//

		// reset surface flux storage as it is used multiple times
		_disc.surfaceFluxParticle[parType].setZero();
		// reset auxiliary g
		g_p.setZero();
		// DG volumne integral: - D c
		parVolumeIntegral(parType, true, conc, g_p);
		// surface integral: M^-1 B [c - c^*]
		parSurfaceIntegral(parType, conc, g_p, strideCell, strideNode, true, comp);
		// auxiliary factor -1
		g_p *= -1.0;
		// => g_p = Dc - M^-1 B [c - c^*]
	}

	// ==========================================================================================================================================================  //
	// ========================================						DG Jacobian							=========================================================  //
	// ==========================================================================================================================================================  //

	// testing purpose
	MatrixXd calcFDJacobian(const SimulationTime simTime, util::ThreadLocalStorage& threadLocalMem, double alpha) {

		// create solution vectors
		VectorXd y = VectorXd::Ones(numDofs());
		VectorXd yDot = VectorXd::Ones(numDofs());
		VectorXd res = VectorXd::Ones(numDofs());
		const double* yPtr = &y[0];
		const double* yDotPtr = &yDot[0];
		double* resPtr = &res[0];
		// create FD jacobian
		MatrixXd Jacobian = MatrixXd::Zero(numDofs(), numDofs());
		// set FD step
		double epsilon = 0.01;

		residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);

		for (int col = 0; col < Jacobian.cols(); col++) {
			Jacobian.col(col) = -(1.0 + alpha) * res;
		}
		/*	 Residual(y+h)	*/
		// state DOFs
		for (int dof = 0; dof < Jacobian.cols(); dof++) {
			y[dof] += epsilon;
			residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);
			y[dof] -= epsilon;
			Jacobian.col(dof) += res;
		}

		// state derivative Jacobian
		for (int dof = 0; dof < Jacobian.cols(); dof++) {
			yDot[dof] += epsilon;
			residualImpl<double, double, double, false>(simTime.t, simTime.secIdx, yPtr, yDotPtr, resPtr, threadLocalMem);
			yDot[dof] -= epsilon;
			Jacobian.col(dof) += alpha * res;
		}

		/*	exterminate numerical noise	and divide by epsilon*/
		for (int i = 0; i < Jacobian.rows(); i++) {
			for (int j = 0; j < Jacobian.cols(); j++) {
				if (std::abs(Jacobian(i, j)) < 1e-10) Jacobian(i, j) = 0.0;
			}
		}
		Jacobian /= epsilon;

		return Jacobian;
	}

	typedef Eigen::Triplet<double> T;

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian for the nodal DG scheme
	*/
	int ConvDispNodalPattern(std::vector<T>& tripletList) {

		Indexer idxr(_disc);

		int sNode = idxr.strideColNode();
		int sCell = idxr.strideColCell();
		int sComp = idxr.strideColComp();
		int offC = idxr.offsetC(); // global jacobian

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
	* @brief sets the sparsity pattern of the convection dispersion Jacobian for the exact integration DG scheme
	*/
	int ConvDispModalPattern(std::vector<T>& tripletList) {

		Indexer idxr(_disc);

		int sNode = idxr.strideColNode();
		int sCell = idxr.strideColCell();
		int sComp = idxr.strideColComp();
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
	 * @brief calculates the particle dispersion jacobian Pattern of the nodal DG scheme for one particle type and bead
	*/
	void calcNodalParticleJacobianPattern(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode) {

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParShell(parType);
		unsigned int sNode = idxr.strideParShell(parType);
		unsigned int sComp = 1u;
		//
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
		unsigned int nNodes = _disc.nParNode[parType];

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < nNodes; j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						tripletList.push_back(T(offset + comp * sComp + i * sNode,
							offset + comp * sComp + j * sNode, 0.0));

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								tripletList.push_back(T(offset + comp * sComp + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode, 0.0));
							}
						}
					}
				}
			}
		}
		else {

			/*			 left boundary cell				*/

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						tripletList.push_back(T(offset + comp * sComp + i * sNode,
							offset + comp * sComp + j * sNode, 0.0));

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								tripletList.push_back(T(offset + comp * sComp + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode, 0.0));
							}
						}
					}
				}
			}

			/*			 right boundary cell				*/

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < nNodes; i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						// handle liquid state
						// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
						// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
						tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
							offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + j * sNode, 0.0));
						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
								tripletList.push_back(T(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell + j * sNode, 0.0));
							}
						}
					}
				}
			}

			/*				inner cells				*/

			for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < nNodes; i++) {
						for (unsigned int j = 0; j < 3 * nNodes; j++) {
							// handle liquid state
							// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
							// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
							tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
								offset + comp * sComp + (cell - 1) * sCell + j * sNode, 0.0));
							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {
								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
									// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
									tripletList.push_back(T(offset + comp * sComp + cell * sCell + i * sNode,
										offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd
										+ (cell - 1) * sCell + j * sNode, 0.0));
								}
							}
						}
					}
				}
			}

		} // if nCells > 1
	}

	/**
	* @brief calculates the number of non zeros for the DG convection dispersion jacobian
	* @detail only dispersion entries are relevant as the convection entries are a subset of these
	*/
	unsigned int calcConvDispNNZ() {

		if (_disc.exactInt) {
			return _disc.nComp * ((3u * _disc.nCol - 2u) * _disc.nNodes * _disc.nNodes + (2u * _disc.nCol - 3u) * _disc.nNodes);
		}
		else {
			return _disc.nComp * (_disc.nCol * _disc.nNodes * _disc.nNodes + 8u * _disc.nNodes);
		}
	}
	unsigned int calcParDispNNZ(int parType) {
		//@TODO!
		if (_disc.exactInt) {
			return _disc.nComp * ((3u * _disc.nParCell[parType] - 2u) * _disc.nParNode[parType] * _disc.nParNode[parType] + (2u * _disc.nParCell[parType] - 3u) * _disc.nParNode[parType]);
		}
		else {
			return _disc.nComp * (_disc.nParCell[parType] * _disc.nParNode[parType] * _disc.nParNode[parType] + 8u * _disc.nParNode[parType]);
		}
	}

	/**
	* @brief sets the sparsity pattern of the convection dispersion Jacobian
	*/
	void setConvDispJacPattern(std::vector<T>& tripletList) {

		// TODO?: convDisp NNZ times two for now, but Convection NNZ < Dispersion NNZ
		tripletList.reserve(2u * calcConvDispNNZ());

		if (_disc.exactInt)
			ConvDispModalPattern(tripletList);
		else
			ConvDispNodalPattern(tripletList);

	}

	void parIsothermPattern_GRM(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode) {

		Indexer idxr(_disc);

		int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });

		// every bound satte might depend on every bound and liquid state
		for (int parNode = 0; parNode < _disc.nParPoints[parType]; parNode++) {
			for (int bnd = 0; bnd < _disc.strideBound[parType]; bnd++) {
				for (int conc = 0; conc < idxr.strideParShell(parType); conc++) {
					// row: jump over previous nodes and liquid states and add current bound state offset
					// col: jump over previous nodes and add current concentration offset (liquid and bound)
					tripletList.push_back(T(offset + parNode * idxr.strideParShell(parType) + idxr.strideParLiquid() + bnd,
						offset + parNode * idxr.strideParShell(parType) + conc, 0.0));
				}
			}
		}
	}

	void parTimeDerJacPattern_GRM(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode) {

		Indexer idxr(_disc);

		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });

		for (unsigned int parNode = 0; parNode < _disc.nParPoints[parType]; parNode++) {
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
					// row: jump over previous nodes add current component offset
					// col: jump over previous nodes, liquid phase and previous bound states
					tripletList.push_back(T(offset + parNode * idxr.strideParShell(parType) + comp,
						offset + parNode * idxr.strideParShell(parType) + idxr.strideParLiquid() + comp + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd, 0.0));
				}
			}
		}
	}

	void setParJacPattern(std::vector<T>& tripletList, unsigned int parType, unsigned int colNode) {

		if (!_disc.parExactInt[parType])
			calcNodalParticleJacobianPattern(tripletList, parType, colNode);
		//else // @TODO?
			//calcModalParticleJacobianPattern(parType, tripletList, mat);

		parTimeDerJacPattern_GRM(tripletList, parType, colNode);

		parIsothermPattern_GRM(tripletList, parType, colNode);

	}

	/**
	 * @brief returns particle diffusion coefficients for one component particle liquid and bound concentrations
	*/
	const Eigen::VectorXd getParDiffComp(unsigned int parType, unsigned int comp, const active* const parDiff, const active* const parSurfDiff) {

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

		diff[0] = static_cast<double>(parDiff[comp]);
		for (int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
			diff[bnd + 1] = static_cast<double>(parSurfDiff[stride[bnd]]);
		}
		return diff;
	}

	/**
	 * @brief analytically calculates the particle dispersion jacobian of the nodal DG scheme for one particle type and bead
	*/
	int calcNodalParticleJacobian(unsigned int parType, unsigned int colNode, const active* const parDiff, const active* const parSurfDiff, const double* const invBetaP) {

		Indexer idxr(_disc);

		// (global) strides
		unsigned int sCell = _disc.nParNode[parType] * idxr.strideParShell(parType);
		unsigned int sNode = idxr.strideParShell(parType);
		unsigned int sComp = 1u;
		//
		unsigned int offset = idxr.offsetCp(ParticleTypeIndex{ parType }, ParticleIndex{ colNode });
		unsigned int nNodes = _disc.nParNode[parType];

		// blocks to compute jacobian
		Eigen::MatrixXd dispBlock;
		double invMap = (2.0 / _disc.deltaR[parType]);
		Eigen::MatrixXd B = MatrixXd::Zero(nNodes, nNodes);
		B(0, 0) = -1.0; B(nNodes - 1, nNodes - 1) = 1.0;

		// special case: one cell -> diffBlock \in R^(nParNodes x nParNodes), GBlock = parPolyDerM
		if (_disc.nParCell[parType] == 1) {

			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0)
				dispBlock = invMap * invMap * (_disc.Dr[parType] - _disc.parInvWeights[parType].asDiagonal() * B) * _disc.parPolyDerM[parType];

			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				dispBlock = MatrixXd::Zero(nNodes, nNodes);

				// reduced system
				dispBlock.block(1, 0, nNodes - 1, nNodes)
					= (_disc.Dr[parType].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1))
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				// inner boundary node
				dispBlock.block(0, 0, 1, nNodes)
					= -(_disc.Ir[parType].segment(1, nNodes - 1).cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* _disc.parPolyDerM[parType].block(1, 0, nNodes - 1, nNodes);

				dispBlock *= invMap * invMap;
			}

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						_globalJac.coeffRef(offset + comp * sComp + i * sNode,
							offset + comp * sComp + j * sNode)
							= -(static_cast<double>(parDiff[comp])) * dispBlock(i, j); // - D_p * (Delta r / 2)^2 * (D_r D - M^-1 B D)

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump oover liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								_globalJac.coeffRef(offset + comp * sComp + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
									= -(static_cast<double>(parSurfDiff[bnd]) * invBetaP[comp]) * dispBlock(i, j); // -  D_s * (1 / Beta_p) * (Delta r / 2)^2 * (D_r D - M^-1 B D)
							}
						}
					}
				}
			}
		}
		else {

			/*			boundary cells			*/
			// initialize dispersion and metric block matrices
			MatrixXd bnd_dispBlock = MatrixXd::Zero(nNodes, 2 * nNodes); // boundary cell specific
			dispBlock = MatrixXd::Zero(nNodes, 3 * nNodes);

			// auxiliary block [ d g(c) / d c ] for left boundary cell
			MatrixXd GBlock_l = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_l.block(0, 0, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_l(nNodes - 1, nNodes - 1) -= 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			GBlock_l(nNodes - 1, nNodes) += 0.5 * _disc.parInvWeights[parType][nNodes - 1];
			// auxiliary block [ d g(c) / d c ] for right boundary cell
			MatrixXd GBlock_r = MatrixXd::Zero(nNodes, nNodes + 1);
			GBlock_r.block(0, 1, nNodes, nNodes) = _disc.parPolyDerM[parType];
			GBlock_r(0, 0) -= 0.5 * _disc.parInvWeights[parType][0];
			GBlock_r(0, 1) += 0.5 * _disc.parInvWeights[parType][0];

			/*			 left boundary cell				*/
			int _cell = 0;
			// numerical flux contribution for right interface of left boundary cell -> d f^*_N / d cp
			MatrixXd bnd_gStarDC = MatrixXd::Zero(nNodes, 2 * nNodes);
			bnd_gStarDC.block(nNodes - 1, 0, 1, nNodes + 1) = GBlock_l.block(nNodes - 1, 0, 1, nNodes + 1);
			bnd_gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 1) += GBlock_r.block(0, 0, 1, nNodes + 1);
			bnd_gStarDC *= 0.5;

			// "standard" computation for slab-shaped particles and spherical, cylindrical particles without core
			if (_parGeomSurfToVol[parType] == _disc.SurfVolRatioSlab || _parCoreRadius[parType] != 0.0) {
				// dispBlock <- invMap^2 * ( D * G_l - M^-1 * B * [G_l - g^*] )
				bnd_dispBlock.block(0, 0, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType]] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_l;
				bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
				bnd_dispBlock *= invMap * invMap;
			}
			else { // special treatment of inner boundary node for spherical and cylindrical particles without particle core

				// inner boundary node
				bnd_dispBlock.block(0, 0, 1, nNodes + 1)
					= -(_disc.Ir[_disc.offsetMetric[parType]].segment(1, nNodes - 1).cwiseProduct(
						_disc.parInvWeights[parType].segment(1, nNodes - 1).cwiseInverse()).cwiseProduct(
							_disc.parPolyDerM[parType].block(1, 0, nNodes - 1, 1))).transpose()
					* GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				// reduced system for remaining nodes
				bnd_dispBlock.block(1, 0, nNodes - 1, nNodes + 1)
					= (_disc.Dr[_disc.offsetMetric[parType]].block(1, 1, nNodes - 1, nNodes - 1)
						- _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1)
						) * GBlock_l.block(1, 0, nNodes - 1, nNodes + 1);

				bnd_dispBlock.block(1, 0, nNodes - 1, 2 * nNodes)
					+= _disc.parInvWeights[parType].segment(1, nNodes - 1).asDiagonal() * B.block(1, 1, nNodes - 1, nNodes - 1) * bnd_gStarDC.block(1, 0, nNodes - 1, 2 * nNodes);

				// mapping
				bnd_dispBlock *= invMap * invMap;
			}

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and go node strides from there for each dispersion block entry
						// col: add component offset and go node strides from there for each dispersion block entry
						_globalJac.coeffRef(offset + comp * sComp + i * sNode,
							offset + comp * sComp + j * sNode)
							= -static_cast<double>(parDiff[comp]) * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add current component offset and go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and go node strides from there for each dispersion block entry
								_globalJac.coeffRef(offset + comp * sComp + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + j * sNode)
									= -static_cast<double>(parSurfDiff[bnd]) * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
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
			// dispBlock <- invMap * ( D_r * G_r - M^-1 * B * [G_r - g^*] )
			bnd_dispBlock.setZero();
			bnd_dispBlock.block(0, nNodes - 1, nNodes, nNodes + 1) = (_disc.Dr[_disc.offsetMetric[parType] + _cell] - _disc.parInvWeights[parType].asDiagonal() * B) * GBlock_r;
			bnd_dispBlock.block(0, 0, nNodes, 2 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * bnd_gStarDC;
			bnd_dispBlock *= invMap * invMap;

			// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
			for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
				for (unsigned int i = 0; i < bnd_dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < bnd_dispBlock.cols(); j++) {
						// handle liquid state
						// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
						// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
						_globalJac.coeffRef(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
							offset + comp * sComp + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
							= -static_cast<double>(parDiff[comp]) * bnd_dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

						// handle surface diffusion of bound states. binding is handled in residualKernel().
						if (_hasSurfaceDiffusion[parType]) {
							for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
								// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
								// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
								_globalJac.coeffRef(offset + comp * sComp + (_disc.nParCell[parType] - 1) * sCell + i * sNode,
									offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd + (_disc.nParCell[parType] - 2) * sCell + j * sNode)
									= -static_cast<double>(parSurfDiff[bnd]) * invBetaP[comp] * bnd_dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
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

			// numerical flux contribution
			MatrixXd gStarDC = MatrixXd::Zero(nNodes, 3 * nNodes);
			gStarDC.block(0, nNodes - 1, 1, nNodes + 2) = GBlock.block(0, 0, 1, nNodes + 2);
			gStarDC.block(0, 0, 1, nNodes + 1) += GBlock.block(nNodes - 1, 1, 1, nNodes + 1);
			gStarDC.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += GBlock.block(nNodes - 1, 0, 1, nNodes + 2);
			gStarDC.block(nNodes - 1, 2 * nNodes - 1, 1, nNodes + 1) += GBlock.block(0, 0, 1, nNodes + 1);
			gStarDC *= 0.5;

			dispBlock.setZero();
			// dispersion block part without metrics
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = -1.0 * _disc.parInvWeights[parType].asDiagonal() * B * GBlock;
			dispBlock.block(0, 0, nNodes, 3 * nNodes) += _disc.parInvWeights[parType].asDiagonal() * B * gStarDC;
			dispBlock *= invMap * invMap;

			for (int cell = 1; cell < _disc.nParCell[parType] - 1; cell++) {
				// add metric part, dependent on current cell
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) += _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock * invMap * invMap;

				// fill the jacobian: add dispersion block for each unbound and bound component, adjusted for the respective coefficients
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// handle liquid state
							// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
							// col: add component offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
							_globalJac.coeffRef(offset + comp * sComp + cell * sCell + i * sNode,
								offset + comp * sComp + (cell - 1) * sCell + j * sNode)
								= -static_cast<double>(parDiff[comp]) * dispBlock(i, j); // dispBlock <- D_p * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]

							// handle surface diffusion of bound states. binding is handled in residualKernel().
							if (_hasSurfaceDiffusion[parType]) {
								for (unsigned int bnd = 0; bnd < _disc.nBound[parType * _disc.nComp + comp]; bnd++) {
									// row: add component offset and jump over previous cells. Go node strides from there for each dispersion block entry
									// col: jump over liquid states, add current bound state offset and jump over previous cells. Go back one cell and go node strides from there for each dispersion block entry
									_globalJac.coeffRef(offset + comp * sComp + cell * sCell + i * sNode,
										offset + idxr.strideParLiquid() + idxr.offsetBoundComp(ParticleTypeIndex{ parType }, ComponentIndex{ comp }) + bnd
										+ (cell - 1) * sCell + j * sNode)
										= -static_cast<double>(parSurfDiff[bnd]) * invBetaP[comp] * dispBlock(i, j); // dispBlock <- D_s * invBeta * [ M^-1 * M_r * G_l +  invMap * ( D * G_l - M^-1 * B * [G_l - g^*] ) ]
								}
							}
						}
					}
				}
				// substract metric part in preparation of next iteration
				dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) -= _disc.Dr[_disc.offsetMetric[parType] + cell] * GBlock * invMap * invMap;
			}

		} // if nCells > 1
		return 0;
	}

	/**
		* @brief analytically calculates the convection dispersion jacobian for the nodal DG scheme
		*/
	int calcConvDispNodalJacobian() {

		Indexer idxr(_disc);

		int sNode = idxr.strideColNode();
		int sCell = idxr.strideColCell();
		int sComp = idxr.strideColComp();
		int offC = idxr.offsetC(); // global jacobian

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
		GBlock *= 2.0 / _disc.deltaZ;

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
		dispBlock *= 2.0 / _disc.deltaZ;

		// insert Blocks to Jacobian inner cells (only for nCells >= 3)
		if (nCells >= 3u) {
			for (unsigned int cell = 1; cell < nCells - 1; cell++) {
				for (unsigned int comp = 0; comp < nComp; comp++) {
					for (unsigned int i = 0; i < dispBlock.rows(); i++) {
						for (unsigned int j = 0; j < dispBlock.cols(); j++) {
							// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each dispersion block entry
							// col: jump over inlet DOFs and previous cells, go back one cell, add component offset and go node strides from there for each dispersion block entry
							_globalJac.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
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
		GBlockBound(0, 1) -= 0.5 * _disc.invWeights[0] * 2.0 / _disc.deltaZ;

		// estimate dispersion block ( j < 0 not needed)
		dispBlock.setZero();
		dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = _disc.polyDerM * GBlockBound;
		dispBlock.block(0, nNodes - 1, 1, nNodes + 2) += -_disc.invWeights[0] * (-GBlockBound.block(0, 0, 1, nNodes + 2)); // G_N,N		i=0, j=-1,...,N+1
		dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound.block(nNodes - 1, 0, 1, nNodes)); // G_i,j+N+1		i=N, j=--1,...,N-1
		dispBlock(nNodes - 1, 2 * nNodes - 1) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound(nNodes - 1, nNodes) + 0.5 * GBlockBound(0, 0)); // G_i,j		i=N, j=N
		dispBlock(nNodes - 1, 2 * nNodes) += _disc.invWeights[nNodes - 1] * (-0.5 * GBlockBound(nNodes - 1, nNodes + 1) + 0.5 * GBlock(0, 1)); // G_i,j		i=N, j=N+1
		dispBlock.block(nNodes - 1, 2 * nNodes + 1, 1, nNodes - 1) += _disc.invWeights[nNodes - 1] * (0.5 * GBlock.block(0, 2, 1, nNodes - 1)); // G_0,j-N-1		i=N, j=N+2,...,2N+1
		dispBlock *= 2.0 / _disc.deltaZ;
		if (nCells != 1u) { // "standard" case
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes; j < dispBlock.cols(); j++) {
						_globalJac.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - nNodes) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}
		else { // special case
			dispBlock.setZero();
			MatrixXd GBlock1Cell = _disc.polyDerM * 2.0 / _disc.deltaZ;
			dispBlock.block(0, nNodes, nNodes, nNodes) = _disc.polyDerM * GBlock1Cell;
			dispBlock.block(0, nNodes, 1, nNodes) += _disc.invWeights[0] * GBlock1Cell.block(0, 0, 1, nNodes);
			dispBlock.block(nNodes - 1, nNodes, 1, nNodes) -= _disc.invWeights[_disc.polyDeg] * GBlock1Cell.block(nNodes - 1, 0, 1, nNodes);
			dispBlock *= 2.0 / _disc.deltaZ;
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = nNodes; j < nNodes * 2u; j++) {
						_globalJac.coeffRef(offC + comp * sComp + i * sNode,
							offC + comp * sComp + (j - nNodes) * sNode)
							= -dispBlock(i, j) * _disc.dispersion[comp];
					}
				}
			}
		}

		/* right cell */
		if (nCells != 1u) { // "standard" case
	   // adjust auxiliary Block [ d g(c) / d c ] for left boundary cell
			GBlockBound(0, 1) += 0.5 * _disc.invWeights[0] * 2.0 / _disc.deltaZ; 	// reverse change from left boundary
			GBlockBound(nNodes - 1, nNodes) += 0.5 * _disc.invWeights[polyDeg] * 2.0 / _disc.deltaZ;

			// estimate dispersion block (only estimation differences to inner cell at N = 0 and j > N not needed)
			dispBlock.block(0, nNodes - 1, nNodes, nNodes + 2) = _disc.polyDerM * GBlockBound;
			dispBlock(0, nNodes - 1) += -_disc.invWeights[0] * (-0.5 * GBlockBound(0, 0) + 0.5 * GBlock(nNodes - 1, nNodes)); // G_N,N		i=0, j=-1
			dispBlock(0, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlockBound(0, 1) + 0.5 * GBlock(nNodes - 1, nNodes + 1)); // G_N,N+1	i=0, j=0
			dispBlock.block(0, nNodes + 1, 1, nNodes) += -_disc.invWeights[0] * (-0.5 * GBlockBound.block(0, 2, 1, nNodes)); // G_i,j		i=0, j=1,...,N+1
			dispBlock.block(0, 0, 1, nNodes - 1) += -_disc.invWeights[0] * (0.5 * GBlock.block(nNodes - 1, 1, 1, nNodes - 1)); // G_N,j+N+1		i=0, j=-N-1,...,-2
			dispBlock.block(nNodes - 1, nNodes - 1, 1, nNodes + 2) += _disc.invWeights[nNodes - 1] * (-GBlockBound.block(nNodes - 1, 0, 1, nNodes + 2)); // G_i,j+N+1		i=N, j=--1,...,N+1
			dispBlock *= 2.0 / _disc.deltaZ;
			// copy *-1 to Jacobian
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < dispBlock.rows(); i++) {
					for (unsigned int j = 0; j < 2 * nNodes; j++) {
						_globalJac.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
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
		convBlock *= 2.0 * _disc.velocity / _disc.deltaZ;

		// special inlet DOF treatment for first cell
		_jacInlet(0, 0) = -convBlock(0, 0); // only first node depends on inlet concentration
		for (unsigned int comp = 0; comp < nComp; comp++) {
			for (unsigned int i = 0; i < convBlock.rows(); i++) {
				for (unsigned int j = 1; j < convBlock.cols(); j++) {
					_globalJac.coeffRef(offC + comp * sComp + i * sNode,
						offC + comp * sComp + (j - 1) * sNode)
						+= -convBlock(i, j);
				}
			}
		}
		for (unsigned int cell = 1; cell < nCells; cell++) {
			for (unsigned int comp = 0; comp < nComp; comp++) {
				for (unsigned int i = 0; i < convBlock.rows(); i++) {
					for (unsigned int j = 0; j < convBlock.cols(); j++) {
						// row: jump over inlet DOFs and previous cells, add component offset and go node strides from there for each convection block entry
						// col: jump over inlet DOFs and previous cells, go back one node, add component offset and go node strides from there for each convection block entry
						_globalJac.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
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
		* @brief analytically calculates the convection dispersion jacobian for the exact integration DG scheme
		*/
	int calcConvDispModalJacobian() {

		Indexer idxr(_disc);

		int sNode = idxr.strideColNode();
		int sCell = idxr.strideColCell();
		int sComp = idxr.strideColComp();
		int offC = idxr.offsetC(); // global jacobian

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
							_globalJac.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + nNodes * sNode + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + (nCells - 2) * sCell + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + (nCells - 1) * sCell + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + 1 * sCell + comp * sComp + i * sNode,
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
					_globalJac.coeffRef(offC + comp * sComp + i * sNode,
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
						_globalJac.coeffRef(offC + cell * sCell + comp * sComp + i * sNode,
							offC + cell * sCell - sNode + comp * sComp + j * sNode)
							-= convBlock(i, j);
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
	int calcStaticAnaBulkJacobian() {

		// DG convection dispersion Jacobian for bulk
		if (_disc.exactInt)
			calcConvDispModalJacobian();
		else
			calcConvDispNodalJacobian();

		if (!_globalJac.isCompressed()) // if matrix lost its compressed storage, the predefined pattern did not fit.
			return 0;

		return 1;
	}

	/**
	* @brief analytically calculates the static (per section) particle jacobian
	* @return 1 if jacobain calculation fits the predefined pattern of the jacobian, 0 if not.
	*/
	int calcStaticAnaParticleDispJacobian(unsigned int parType, unsigned int colNode, const active* const parDiff, const active* const parSurfDiff, const double* const invBetaP) {

		// DG particle dispersion Jacobian
		if (_disc.parExactInt[parType])
			throw std::invalid_argument("exact integration Particle Jacobian not implemented yet");
		//calcModalParticleJacobian(parType, _jacP[parType * colNode]);
		else
			calcNodalParticleJacobian(parType, colNode, parDiff, parSurfDiff, invBetaP);

		return _globalJac.isCompressed(); // if matrix lost its compressed storage, the calculation did not fit the pre-defined pattern.
	}


	void setJacobianPattern_GRM(SparseMatrix<double, RowMajor>& globalJ) {

		Indexer idxr(_disc);

		std::vector<T> tripletList;
		// reserve space for all entries
		int bulkEntries = calcConvDispNNZ();
		// particle
		int addTimeDer = 0; // additional time derivative entries: bound states in particle dispersion equation
		int isothermNNZ = 0;
		int particleEntries = 0;
		for (int type = 0; type < _disc.nParType; type++) {
			isothermNNZ = (idxr.strideParShell(type)) * _disc.nParPoints[type] * _disc.strideBound[type]; // every bound satte might depend on every bound and liquid state
			addTimeDer = _disc.nParPoints[type] * _disc.strideBound[type];
			particleEntries += calcParDispNNZ(type) + addTimeDer + isothermNNZ;
		}
		// flux @TODO
		int nBCSurfDiff = 0; // Boundary condition surface diffusion
		//for (int type = 0; type < _disc.nParType; type++) {
		//	if (_disc.hasSurfDiff(type))
		//		nBCSurfDiff += _disc.strideBound[type];
		//}
		int fluxEntries = 4 * _disc.nParType * _disc.nPoints * _disc.nComp + nBCSurfDiff;

		tripletList.reserve(fluxEntries + bulkEntries + particleEntries);

		// NOTE: inlet and jacF flux jacobian is set in calc jacobian function (identity matrices)

		// bulk jacobian
		setConvDispJacPattern(tripletList);

		// particle jacobian (including isotherm and time derivative)
		for (int colNode = 0; colNode < _disc.nPoints; colNode++) {
			for (int type = 0; type < _disc.nParType; type++) {
				setParJacPattern(tripletList, type, colNode);
			}
		}

		// fluxJacobians
		for (unsigned int type = 0; type < _disc.nParType; type++) {
			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++) {
				for (unsigned int comp = 0; comp < _disc.nComp; comp++) {
					// add Cl on F entries
					// row: add bulk offset, jump over previous nodes and components
					// col: add flux offset to current parType, jump over previous nodes and components
					tripletList.push_back(T(idxr.offsetC() + colNode * idxr.strideColNode() + comp * idxr.strideColComp(),
						idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp, 0.0));
					// add F on Cl entries
					// row: add flux offset to current parType, jump over previous nodes and components
					// col: add bulk offset, jump over previous nodes and components
					tripletList.push_back(T(idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp,
						idxr.offsetC() + colNode * idxr.strideColNode() + comp * idxr.strideColComp(), 0.0));
					// add Cp on F entries
					// row: add particle offset to current parType, jump over previous particles, go to last node and add component offset
					// col: add flux offset to current component, jump over previous nodes and components
					tripletList.push_back(T(idxr.offsetCp(ParticleTypeIndex{ type }) + colNode * _disc.nParPoints[type] * idxr.strideParShell(type)
						+ (_disc.nParPoints[type] - 1) * idxr.strideParShell(type) + comp * idxr.strideParComp(),
						idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp, 0.0));

					//// @TODO: if surface diffusion with quasi stationary binding: add dependence of c^s on f !
					//if (bindingQuasiStationary(comp) && surface diffusion) {
					//	for (int bnd = 0; bnd < _disc.nBound[type * _disc.nComp + comp]; bnd++) {
					//		// add Cs on F entries
					//		// row: add particle offset to current parType, jump over previous particles and particle liquid and add boundstateOffset
					//		// col: add flux offset to current component, jump over previous nodes and components
					//		globalList.push_back(T(_disc.offsetCp(type) + colNode * _disc.nParPoints[type] * _disc.strideParShell(type) + _disc.strideParLiquid() + _disc.boundOffset(type, comp) + bnd,
					//			_disc.offsetJf(type) + colNode * _disc.nComp + comp, 3.0));
					//	}
					//}

					// add F on Cp entries
					// row: add flux offset to current parType, jump over previous nodes and components
					// col: add particle offset to current parType, jump over previous particles and components
					tripletList.push_back(T(idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp,
						idxr.offsetCp(ParticleTypeIndex{ type }) + colNode * _disc.nParPoints[type] * idxr.strideParShell(type)
						+ (_disc.nParPoints[type] - 1) * idxr.strideParShell(type) + comp * idxr.strideParComp(), 0.0));

					// add F on F entries (identity matrix)
					// row: add flux offset to current parType, jump over previous nodes and components
					// col: add flux offset to current parType, jump over previous nodes and components
					tripletList.push_back(T(idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp,
						idxr.offsetJf(ParticleTypeIndex{ type }) + colNode * _disc.nComp + comp, 0.0));
				}
			}
		}

		// Note: flux jacobian (identity matrix) is handled in calc jacobian function

		globalJ.setFromTriplets(tripletList.begin(), tripletList.end());
	}

	int calcFluxJacobians(unsigned int secIdx) {

		Indexer idxr(_disc);

		for (unsigned int type = 0; type < _disc.nParType; type++) {

			// Ordering of diffusion:
			// sec0type0comp0, sec0type0comp1, sec0type0comp2, sec0type1comp0, sec0type1comp1, sec0type1comp2,
			// sec1type0comp0, sec1type0comp1, sec1type0comp2, sec1type1comp0, sec1type1comp1, sec1type1comp2, ...
			active const* const filmDiff = getSectionDependentSlice(_filmDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

			linalg::BandedEigenSparseRowIterator jacClF(_globalJac, idxr.offsetC());
			linalg::BandedEigenSparseRowIterator jacFCl(_globalJac, idxr.offsetJf(ParticleTypeIndex{ type }));
			linalg::BandedEigenSparseRowIterator jacCpF(_globalJac, idxr.offsetCp(ParticleTypeIndex{ type }) + (_disc.nParPoints[type] - 1) * idxr.strideParShell(type));
			linalg::BandedEigenSparseRowIterator jacFCp(_globalJac, idxr.offsetJf(ParticleTypeIndex{ type }));
			// @TODO: quasi-stationary binding, surfacediffusion BC
			linalg::BandedEigenSparseRowIterator jacFF(_globalJac, idxr.offsetJf(ParticleTypeIndex{ type }));

			for (unsigned int colNode = 0; colNode < _disc.nPoints; colNode++, jacCpF += _disc.strideBound[type] + (_disc.nParPoints[type] - 1) * idxr.strideParShell(type))
			{
				for (unsigned int comp = 0; comp < _disc.nComp; comp++, ++jacClF, ++jacFCl, ++jacCpF, ++jacFCp, ++jacFF) {
					// add Cl on F entries
					// row: already at bulk phase. already at current node and component.
					// col: add offset to Flux of current type. already at current node and component.
					jacClF[idxr.offsetJf(ParticleTypeIndex{ type }) - idxr.offsetC()] = (1.0 - static_cast<double>(_colPorosity)) / static_cast<double>(_colPorosity)
																						* _parGeomSurfToVol[type] / static_cast<double>(_parRadius[type])
																						* _parTypeVolFrac[type + colNode * _disc.nParType].getValue();

					// add F on Cl entries
					// row: already at flux of current parType. already at current node and component.
					// col: go back to bulk phase. already at current node and component.
					jacFCl[-idxr.offsetJf(ParticleTypeIndex{ type }) + idxr.offsetC()] = -static_cast<double>(filmDiff[comp]);

					// add Cp on F entries (inexact integration scheme)
					// row: already at particle. already at current node and liquid state.
					// col: go to flux of current parType. jump over previous colNodes and add component offset
					jacCpF[idxr.offsetJf(ParticleTypeIndex{ type }) - jacCpF.row() + colNode * _disc.nComp + comp]
						= -2.0 / _disc.deltaR[type] * _disc.parInvWeights[type][0] / static_cast<double>(_parPorosity[type]) / static_cast<double>(_poreAccessFactor[type * _disc.nComp + comp]);

					//// @TODO: add Cs on F entries for quasi-stationary binding, surfacediffusion BC
					// if(){
					//// row:
					//// col:
					// jacCpF[] = ;
					// }

					// add F on Cp entries
					// row: already at flux of current parType. already at current node and component.
					// col: go back to current particle type. jump over previous particles, go to last node and add component offset
					jacFCp[-jacFCp.row() + idxr.offsetCp(ParticleTypeIndex{ type }) + colNode * _disc.nParPoints[type] * idxr.strideParShell(type)
						+ (_disc.nParPoints[type] - 1) * idxr.strideParShell(type) + comp * idxr.strideParComp()]
						= static_cast<double>(filmDiff[comp]);

					// add F on F entries (identity matrix)
					jacFF[0] = 1.0;
				}
			}
		}

		return 0;
	}

	int calcStaticAnaJacobian_GRM(unsigned int secIdx) {

		Indexer idxr(_disc);
		// inlet and bulk jacobian
		calcStaticAnaBulkJacobian();

		// particle jacobian (without isotherm, which is handled in residualKernel)
		for (int colNode = 0; colNode < _disc.nPoints; colNode++) {
			for (int type = 0; type < _disc.nParType; type++) {

				// Prepare parameters
				const active* const parDiff = getSectionDependentSlice(_parDiffusion, _disc.nComp * _disc.nParType, secIdx) + type * _disc.nComp;

				// Ordering of particle surface diffusion:
				// bnd0comp0, bnd0comp1, bnd0comp2, bnd1comp0, bnd1comp1, bnd1comp2
				const active* const  parSurfDiff = getSectionDependentSlice(_parSurfDiffusion, _disc.strideBound[_disc.nParType], secIdx) + _disc.nBoundBeforeType[type];

				double* invBetaP = new double[_disc.nComp];
				for (int comp = 0; comp < _disc.nComp; comp++) {
					invBetaP[comp] = (1.0 - static_cast<double>(_parPorosity[type])) / (static_cast<double>(_poreAccessFactor[_disc.nComp * type + comp]) * static_cast<double>(_parPorosity[type]));
				}

				calcStaticAnaParticleDispJacobian(type, colNode, parDiff, parSurfDiff, invBetaP);
				
			}
		}

		// fluxJacobians J_FC, J_CF
		calcFluxJacobians(secIdx);
		// J_F (identity matrix)
		linalg::BandedEigenSparseRowIterator jac(_globalJac, idxr.offsetJf());
		for (int flux = 0; flux < _disc.nPoints * _disc.nParType; flux++) {
			jac[0] = 1.0;
		}

		return _globalJac.isCompressed(); // check if the jacobian estimation fits the pattern
	}

};

} // namespace model
} // namespace cadet

#endif  // LIBCADET_GENERALRATEMODELDG_HPP_
