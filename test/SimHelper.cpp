// =============================================================================
//  CADET - The Chromatography Analysis and Design Toolkit
//  
//  Copyright © 2008-2017: The CADET Authors
//            Please see the AUTHORS and CONTRIBUTORS file.
//  
//  All rights reserved. This program and the accompanying materials
//  are made available under the terms of the GNU Public License v3.0 (or, at
//  your option, any later version) which accompanies this distribution, and
//  is available at http://www.gnu.org/licenses/gpl.html
// =============================================================================

#include "SimHelper.hpp"

#include <sstream>
#include <iomanip>

namespace
{
	class GroupScope
	{
	public:
		GroupScope(cadet::JsonParameterProvider& jpp, unsigned int idxUnit) : _jpp(jpp)
		{
			_active = jpp.exists("model");
			if (_active)
			{
				std::ostringstream ss;
				_jpp.pushScope("model");
				ss << "unit_" << std::setfill('0') << std::setw(3) << idxUnit;
				_jpp.pushScope(ss.str());
			}
		}

		GroupScope(cadet::JsonParameterProvider& jpp) : GroupScope(jpp, 0) { }

		~GroupScope()
		{
			if (_active)
			{
				_jpp.popScope();
				_jpp.popScope();
			}
		}

	private:
		bool _active;
		cadet::JsonParameterProvider& _jpp;
	};
}

namespace cadet
{

namespace test
{

	void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c, const std::vector<double>& q, double v)
	{
		GroupScope gs(jpp);

		jpp.set("INIT_C", c);
		jpp.set("INIT_VOLUME", v);
		if (!q.empty())
			jpp.set("INIT_Q", q);
	}

	void setInitialConditions(cadet::JsonParameterProvider& jpp, const std::vector<double>& c, const std::vector<double>& cp, const std::vector<double>& q)
	{
		GroupScope gs(jpp);

		jpp.set("INIT_C", c);
		if (!cp.empty())
			jpp.set("INIT_CP", cp);
		if (!q.empty())
			jpp.set("INIT_Q", q);
	}

	void setFlowRates(cadet::JsonParameterProvider& jpp, unsigned int secIdx, double in, double out, double filter)
	{
		GroupScope gs(jpp);

		std::vector<double> frf = jpp.getDoubleArray("FLOWRATE_FILTER");
		if (frf.size() <= secIdx)
			std::fill_n(std::back_inserter(frf), frf.size() - secIdx + 1, 0.0);

		frf[secIdx] = filter;
		jpp.set("FLOWRATE_FILTER", frf);

		jpp.popScope();
		jpp.pushScope("connections");

		std::ostringstream ss;
		ss << "switch_" << std::setfill('0') << std::setw(3) << secIdx;
		jpp.pushScope(ss.str());

		std::vector<double> con = jpp.getDoubleArray("CONNECTIONS");
		con[4] = in;
		con[9] = out;
		jpp.set("CONNECTIONS", con);

		jpp.popScope();
	}

	void setInletProfile(cadet::JsonParameterProvider& jpp, unsigned int secIdx, unsigned int comp, double con, double lin, double quad, double cub)
	{
		GroupScope gs(jpp, 1);

		std::ostringstream ss;
		ss << "sec_" << std::setfill('0') << std::setw(3) << secIdx;
		jpp.pushScope(ss.str());

		std::vector<double> cCon = jpp.getDoubleArray("CONST_COEFF");
		cCon[comp] = con;
		jpp.set("CONST_COEFF", cCon);

		std::vector<double> cLin = jpp.getDoubleArray("LIN_COEFF");
		cLin[comp] = lin;
		jpp.set("LIN_COEFF", cLin);

		std::vector<double> cQuad = jpp.getDoubleArray("QUAD_COEFF");
		cQuad[comp] = quad;
		jpp.set("QUAD_COEFF", cQuad);

		std::vector<double> cCube = jpp.getDoubleArray("CUBE_COEFF");
		cCube[comp] = cub;
		jpp.set("CUBE_COEFF", cCube);

		jpp.popScope();
	}

	void setSectionTimes(cadet::JsonParameterProvider& jpp, const std::vector<double>& secTimes)
	{
		jpp.pushScope("solver");
		jpp.pushScope("sections");

		jpp.set("SECTION_TIMES", secTimes);

		jpp.popScope();
		jpp.popScope();
	}

	void addBoundStates(cadet::JsonParameterProvider& jpp, const std::vector<int>& nBound, double porosity)
	{
		GroupScope gs(jpp);

		jpp.set("NBOUND", nBound);
		jpp.set("POROSITY", porosity);
	}

	void addLinearBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA, const std::vector<double>& kD)
	{
		GroupScope gs(jpp);

		jpp.set("ADSORPTION_MODEL", "LINEAR");

		jpp.addScope("adsorption");
		jpp.pushScope("adsorption");

		jpp.set("IS_KINETIC", kinetic);
		jpp.set("LIN_KA", kA);
		jpp.set("LIN_KD", kD);

		jpp.popScope();
	}

	void addLangmuirBindingModel(cadet::JsonParameterProvider& jpp, bool kinetic, const std::vector<double>& kA, const std::vector<double>& kD, const std::vector<double>& qMax)
	{
		GroupScope gs(jpp);

		jpp.set("ADSORPTION_MODEL", "MULTI_COMPONENT_LANGMUIR");

		jpp.addScope("binding");
		jpp.pushScope("binding");

		jpp.set("IS_KINETIC", kinetic);
		jpp.set("MCL_KA", kA);
		jpp.set("MCL_KD", kD);
		jpp.set("MCL_QMAX", qMax);

		jpp.popScope();
	}

	void addSensitivity(cadet::JsonParameterProvider& jpp, const std::string& name, const ParameterId& id, double absTol)
	{
		jpp.addScope("sensitivity");
		jpp.pushScope("sensitivity");

		jpp.set("SENS_METHOD", "ad1");
		int sensIdx = 0;
		if (jpp.exists("NSENS"))
		{
			sensIdx = jpp.getInt("NSENS");
		}
		jpp.set("NSENS", sensIdx + 1);

		std::ostringstream ss;
		ss << "param_" << std::setfill('0') << std::setw(3) << sensIdx;
		jpp.addScope(ss.str());
		jpp.pushScope(ss.str());

		jpp.set("SENS_UNIT", std::vector<int>(1, id.unitOperation));
		jpp.set("SENS_NAME", std::vector<std::string>(1, name));
		jpp.set("SENS_COMP", std::vector<int>(1, id.component));
		jpp.set("SENS_BOUNDPHASE", std::vector<int>(1, id.boundPhase));
		jpp.set("SENS_REACTION", std::vector<int>(1, id.reaction));
		jpp.set("SENS_SECTION", std::vector<int>(1, id.section));
		jpp.set("SENS_FACTOR", std::vector<double>(1, 1.0));
		jpp.set("SENS_ABSTOL", absTol);

		jpp.popScope();
		jpp.popScope();
	}

	void returnSensitivities(cadet::JsonParameterProvider& jpp, UnitOpIdx unit, bool inlet)
	{
		jpp.pushScope("return");

		std::ostringstream ss;
		ss << "unit_" << std::setfill('0') << std::setw(3) << unit;
		jpp.pushScope(ss.str());

		jpp.set("WRITE_SENS_INLET", inlet);
		jpp.set("WRITE_SENS_OUTLET", true);
		jpp.set("WRITE_SENS_VOLUME", true);
		jpp.popScope();
		jpp.popScope();
	}

	void disableSensitivityErrorTest(cadet::JsonParameterProvider& jpp, bool isDisabled)
	{
		jpp.pushScope("solver");
		jpp.pushScope("time_integrator");
		jpp.set("ERRORTEST_SENS", !isDisabled);
		jpp.popScope();
		jpp.popScope();
	}

} // namespace test
} // namespace cadet