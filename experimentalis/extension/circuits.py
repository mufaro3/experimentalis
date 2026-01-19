"""
This extension module is used for dealing with circuits, namely their
response curves.
"""

from experimentalis.models import Model

class RCResponseModel(Model):
    """
    A response model for Resistor-Capacitor (RC) circuits in the frequency domain.
    
    .. math::
        :label: eq-rc-response

        V(f) = V_0 / \sqrt{1 + (2\pi f \tau)^2} + V_C

    :param amplitude: Amplitude :math:`V_0`
    :param time_constant: Frequency :math:`\\tau = RC`
    :param offset: Phase :math:`V_C`

    :type amplitude: float
    :type time_constant: float
    :type offset: float
    """
    def __init__(self, amplitude, time_constant, offset):
        def fit_function(x, amplitude, time_constant, offset):
            return amplitude / np.sqrt(1 + (2 * np.pi * x * time_constant) ** 2) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Offset'],
            param_values=np.array([amplitude, time_constant, offset]),
            param_uncerts=np.array([-1, -1, -1])
        )

class LRCResponseModel(Model):
    """
    A response model for Inductor-Resistor-Capacitor (LRC) circuits in the frequency domain.
    
    .. math::
        :label: eq-lrc-response

        V(f) = V_0/\sqrt{1 + \left(\frac{2\pi}{\gamma f}\right)^2 (f^2 - f^2_0)^2}
    
    :param amplitude: Amplitude :math:`V_0`
    :param damping_rate: Damping rate :math:`\\gamma = R/L`, which determines the resonance linewidth and decay rate of oscillations in the time domain.
    :param resonant_frequency: Phase :math:`f_0`

    :type amplitude: float
    :type damping_rate: float
    :type resonant_frequency: float
    """
    def __init__(self, amplitude, damping_rate, resonant_frequency):
        def fit_function(x, amplitude, damping_rate, resonant_frequency):
            return amplitude / np.sqrt(1 + (2 * np.pi / (gamma * x)) ** 2 *(x**2-resonant_frequency ** 2) ** 2)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Damping Rate', 'Resonant Frequency'],
            param_values=np.array([amplitude, time_constant, resonant_frequency]),
            param_uncerts=np.array([-1, -1, -1])
        )
