
using Leap71.ShapeKernel;
using Leap71.LatticeLibraryExamples;
using PicoGK;

string strOutputFolder = "/workspace/LEAP71_version_KBE/PicoGK_Examples-main/Examples";


try
{
    PicoGK.Library.Go(
        0.5f,
        Leap71.CoolCube.HelixHeatX.Task,
        strOutputFolder);
}
catch (Exception e)
{
    Console.WriteLine("Failed to run Task.");
    Console.WriteLine(e.ToString());
}