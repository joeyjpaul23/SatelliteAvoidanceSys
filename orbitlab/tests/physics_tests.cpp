#include "orbitlab/physics.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace orbitlab;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    const Elements source{radius_earth+700.0,0.012,rad(63.4),rad(127.0),rad(41.0),rad(238.0)};
    const State state=from_elements(source);
    const Elements recovered=to_elements(state);
    require(std::abs(source.a_km-recovered.a_km)<1e-7,"semi-major axis round-trip");
    require(std::abs(source.e-recovered.e)<1e-10,"eccentricity round-trip");
    require(std::abs(source.i_rad-recovered.i_rad)<1e-10,"inclination round-trip");

    const Vehicle satellite{"TEST","Invariant test","payload",500.0,4.0,2.2,1.3,state};
    const ForceModel kepler{false,false,false};
    const auto orbit=propagate(satellite,kepler,period_s(state),5.0,60.0);
    double max_energy_error=0.0;
    const double e0=orbit.front().energy;
    for (const auto& sample:orbit) max_energy_error=std::max(max_energy_error,std::abs(sample.energy-e0));
    require(max_energy_error<1e-9,"RK4 two-body energy invariant");
    require(norm(accel_j2({radius_earth+700.0,0,0}))>1e-6,"J2 acceleration is active");
    require(density_kg_m3(500.0)<density_kg_m3(300.0),"density falls with altitude");
    require(cylindrical_eclipse({-7000,0,0},{1,0,0}),"cylindrical shadow geometry");
    std::cout << "physics invariants: PASS (max |dE|=" << max_energy_error << " km^2/s^2)\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "physics invariants: FAIL: " << error.what() << '\n';
    return 1;
  }
}
