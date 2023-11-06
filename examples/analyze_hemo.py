import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from aorta_lib import utils
from aorta_lib import vmtk_helper
from pathlib import Path

hemo_output_dir = Path('../aorta_lib/hemodyn/odir_noLATIN1_evaluate')
samples = hemo_output_dir.glob('*.vtp')
analysis_odir = Path(str(hemo_output_dir)+'_vmtk')
analysis_odir.mkdir()

for sample in samples:
    print(sample)
    m = pv.read(sample)
    cl, m_metrics = vmtk_helper.vmtk_analysis(m)
    cl.save(analysis_odir / (sample.stem+'_cl.vtp'))
    m_metrics.save(analysis_odir / (sample.stem+'_metrics.vtp'))


#nbins = 31
#long = m_metrics['AbscissaMetric']
#bins_true, av_true = theta_average(long, m_metrics['tawss'], nbins)
##bins_rom, av_rom = theta_average(long, m_metrics['tawss_ROM'], nbins)
#bins_rom, av_rom = theta_average(long, m_metrics['tawss_ROM_val'], nbins)

#plt.plot(av_true)
#plt.plot(av_rom)
#plt.show()
