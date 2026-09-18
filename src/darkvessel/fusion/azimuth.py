"""Azimuth-shift correction.

SAR reads along-track (azimuth) position from the Doppler history of the echo, so a target
moving during the synthetic-aperture integration is drawn displaced along the satellite's
ground track, by an amount proportional to its velocity component along the line of sight.
Uncorrected, this alone puts a fast declared vessel hundreds of metres from its detection and
reports it dark. See the README's "How it works" for where this sits in the pipeline.
"""

from dataclasses import dataclass
from math import cos, radians, sin


@dataclass(frozen=True)
class Geometry:
    """The acquisition geometry needed to predict where the radar draws a moving target.

    `heading_deg` is the satellite's ground-track direction, clockwise from north.
    `incidence_deg` is the radar incidence angle at the scene centre. `shift_per_mps` is the
    azimuth displacement, in metres, per metre-per-second of line-of-sight velocity —
    physically slant_range / satellite_velocity; ~113 s is typical for Sentinel-1 IW. The
    satellite is assumed right-looking, standard for Sentinel-1.
    """

    heading_deg: float
    incidence_deg: float
    shift_per_mps: float = 113.0

    def displacement(
        self, velocity_east_ms: float, velocity_north_ms: float, latitude: float
    ) -> tuple[float, float]:
        """Return the (east_m, north_m) displacement this velocity causes in the drawn image.

        `latitude` is accepted for interface stability — a future refinement (e.g. correcting
        for meridian convergence away from the scene centre) would need it — but does not
        change today's flat-earth approximation.
        """
        del latitude
        heading = radians(self.heading_deg)
        incidence = radians(self.incidence_deg)
        flight_east, flight_north = sin(heading), cos(heading)
        range_east, range_north = cos(heading), -sin(heading)
        line_of_sight_velocity = (
            velocity_east_ms * range_east + velocity_north_ms * range_north
        ) * sin(incidence)
        shift_m = self.shift_per_mps * line_of_sight_velocity
        return shift_m * flight_east, shift_m * flight_north
