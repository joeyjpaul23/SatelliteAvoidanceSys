#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace orbitlab {

constexpr double pi = 3.1415926535897932384626433832795;
constexpr double mu_earth = 398600.4418;          // km^3 s^-2, EGM-96
constexpr double radius_earth = 6378.1363;        // km, equatorial
constexpr double j2_earth = 1.08262668e-3;
constexpr double omega_earth = 7.2921150e-5;      // rad s^-1
constexpr double au_km = 149597870.7;
constexpr double solar_pressure = 4.56e-6;        // N m^-2 at 1 AU

struct Vec3 {
  double x{};
  double y{};
  double z{};

  constexpr Vec3 operator+(const Vec3& v) const { return {x + v.x, y + v.y, z + v.z}; }
  constexpr Vec3 operator-(const Vec3& v) const { return {x - v.x, y - v.y, z - v.z}; }
  constexpr Vec3 operator-() const { return {-x, -y, -z}; }
  constexpr Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
  constexpr Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
  Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }
inline double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline double norm2(const Vec3& v) { return dot(v, v); }
inline double norm(const Vec3& v) { return std::sqrt(norm2(v)); }
inline Vec3 normalized(const Vec3& v) { const double n = norm(v); return n > 0.0 ? v / n : Vec3{}; }
inline double rad(double degrees) { return degrees * pi / 180.0; }
inline double deg(double radians) { return radians * 180.0 / pi; }

struct State {
  Vec3 r; // ECI km
  Vec3 v; // ECI km/s
};

inline State operator+(const State& a, const State& b) { return {a.r+b.r, a.v+b.v}; }
inline State operator*(const State& a, double s) { return {a.r*s, a.v*s}; }

struct Elements {
  double a_km{};
  double e{};
  double i_rad{};
  double raan_rad{};
  double argp_rad{};
  double nu_rad{};
};

inline double wrap_2pi(double x) {
  x = std::fmod(x, 2.0*pi);
  return x < 0.0 ? x + 2.0*pi : x;
}

inline State from_elements(const Elements& el) {
  const double p = el.a_km * (1.0 - el.e*el.e);
  const double cnu = std::cos(el.nu_rad), snu = std::sin(el.nu_rad);
  const Vec3 rp{p*cnu/(1.0+el.e*cnu), p*snu/(1.0+el.e*cnu), 0.0};
  const double q = std::sqrt(mu_earth/p);
  const Vec3 vp{-q*snu, q*(el.e+cnu), 0.0};
  const double O=el.raan_rad, i=el.i_rad, w=el.argp_rad;
  const double cO=std::cos(O), sO=std::sin(O), ci=std::cos(i), si=std::sin(i);
  const double cw=std::cos(w), sw=std::sin(w);
  const std::array<std::array<double,3>,3> Q{{
    {{cO*cw-sO*sw*ci, -cO*sw-sO*cw*ci, sO*si}},
    {{sO*cw+cO*sw*ci, -sO*sw+cO*cw*ci, -cO*si}},
    {{sw*si, cw*si, ci}}
  }};
  auto rotate = [&](const Vec3& a) { return Vec3{
    Q[0][0]*a.x+Q[0][1]*a.y+Q[0][2]*a.z,
    Q[1][0]*a.x+Q[1][1]*a.y+Q[1][2]*a.z,
    Q[2][0]*a.x+Q[2][1]*a.y+Q[2][2]*a.z}; };
  return {rotate(rp), rotate(vp)};
}

inline Elements to_elements(const State& s) {
  const double rmag=norm(s.r), vmag=norm(s.v);
  const Vec3 h=cross(s.r,s.v), n=cross({0,0,1},h);
  const Vec3 evec=cross(s.v,h)/mu_earth - s.r/rmag;
  const double hmag=norm(h), nmag=norm(n), e=norm(evec);
  const double energy=0.5*vmag*vmag-mu_earth/rmag;
  Elements out{-mu_earth/(2.0*energy), e, std::acos(std::clamp(h.z/hmag,-1.0,1.0)), 0, 0, 0};
  if (nmag > 1e-12) out.raan_rad=wrap_2pi(std::atan2(n.y,n.x));
  if (nmag > 1e-12 && e > 1e-10)
    out.argp_rad=wrap_2pi(std::atan2(dot(cross(n,evec),h)/(nmag*e*hmag),dot(n,evec)/(nmag*e)));
  if (e > 1e-10)
    out.nu_rad=wrap_2pi(std::atan2(dot(cross(evec,s.r),h)/(e*rmag*hmag),dot(evec,s.r)/(e*rmag)));
  else if (nmag > 1e-12)
    out.nu_rad=wrap_2pi(std::atan2(dot(cross(n,s.r),h)/(nmag*rmag*hmag),dot(n,s.r)/(nmag*rmag)));
  return out;
}

struct Vehicle {
  std::string id;
  std::string name;
  std::string kind;
  double mass_kg{};
  double area_m2{};
  double cd{2.2};
  double cr{1.3};
  State initial;
};

struct ForceModel {
  bool j2{true};
  bool drag{true};
  bool srp{true};
};

inline double density_kg_m3(double altitude_km) {
  // US Standard Atmosphere-style exponential bands, adequate for force sensitivity work.
  struct Band { double h, rho, H; };
  static constexpr std::array<Band,8> table{{
    {120,2.438e-8,9.473},{150,2.070e-9,22.523},{200,2.789e-10,37.105},
    {250,7.248e-11,45.546},{300,2.418e-11,53.628},{400,3.725e-12,58.515},
    {500,6.967e-13,63.822},{600,1.454e-13,71.835}}};
  const Band* b=&table.front();
  for (const auto& candidate:table) if (altitude_km>=candidate.h) b=&candidate;
  return b->rho*std::exp(-(altitude_km-b->h)/b->H);
}

inline Vec3 accel_two_body(const Vec3& r) {
  const double d=norm(r); return (-mu_earth/(d*d*d))*r;
}

inline Vec3 accel_j2(const Vec3& r) {
  const double d=norm(r), d2=d*d, z2=r.z*r.z;
  const double k=1.5*j2_earth*mu_earth*radius_earth*radius_earth/std::pow(d,5);
  return {k*r.x*(5.0*z2/d2-1.0), k*r.y*(5.0*z2/d2-1.0), k*r.z*(5.0*z2/d2-3.0)};
}

inline bool cylindrical_eclipse(const Vec3& r, const Vec3& sun_dir) {
  const double axial=dot(r,sun_dir);
  const Vec3 perpendicular=r-axial*sun_dir;
  return axial<0.0 && norm(perpendicular)<radius_earth;
}

inline Vec3 acceleration(const State& s, const Vehicle& vehicle, const ForceModel& model) {
  Vec3 a=accel_two_body(s.r);
  if (model.j2) a+=accel_j2(s.r);
  if (model.drag) {
    const double altitude=norm(s.r)-radius_earth;
    const Vec3 atmosphere_v=cross({0,0,omega_earth},s.r);
    const Vec3 rel=s.v-atmosphere_v;
    const double ballistic=vehicle.cd*vehicle.area_m2/vehicle.mass_kg;
    // rho kg/m3, velocity km/s -> acceleration km/s2 conversion is 1000.
    a+=(-0.5*density_kg_m3(altitude)*ballistic*norm(rel)*rel*1000.0);
  }
  const Vec3 sun_dir=normalized(Vec3{0.9175,0.3651,0.1583});
  if (model.srp && !cylindrical_eclipse(s.r,sun_dir)) {
    const double srp_km_s2=solar_pressure*vehicle.cr*vehicle.area_m2/vehicle.mass_kg/1000.0;
    a+=(-srp_km_s2)*sun_dir;
  }
  return a;
}

inline State derivative(const State& s, const Vehicle& vehicle, const ForceModel& model) {
  return {s.v,acceleration(s,vehicle,model)};
}

inline State rk4_step(const State& s, double dt, const Vehicle& vehicle, const ForceModel& model) {
  const State k1=derivative(s,vehicle,model);
  const State k2=derivative(s+k1*(0.5*dt),vehicle,model);
  const State k3=derivative(s+k2*(0.5*dt),vehicle,model);
  const State k4=derivative(s+k3*dt,vehicle,model);
  return s+(k1+k2*2.0+k3*2.0+k4)*(dt/6.0);
}

inline double specific_energy(const State& s) { return 0.5*norm2(s.v)-mu_earth/norm(s.r); }
inline double period_s(const State& s) {
  const double a=to_elements(s).a_km; return 2.0*pi*std::sqrt(a*a*a/mu_earth);
}

struct Sample {
  double t_s{};
  State state;
  Elements elements;
  double energy{};
  double density{};
};

inline std::vector<Sample> propagate(const Vehicle& vehicle, const ForceModel& model,
                                     double duration_s, double integrator_step_s,
                                     double output_step_s) {
  std::vector<Sample> samples;
  State state=vehicle.initial;
  double t=0.0, next_output=0.0;
  while (t<=duration_s+1e-9) {
    if (t+1e-9>=next_output) {
      samples.push_back({t,state,to_elements(state),specific_energy(state),density_kg_m3(norm(state.r)-radius_earth)});
      next_output+=output_step_s;
    }
    const double dt=std::min(integrator_step_s,duration_s-t);
    if (dt<=0.0) break;
    state=rk4_step(state,dt,vehicle,model);
    t+=dt;
  }
  return samples;
}

} // namespace orbitlab
