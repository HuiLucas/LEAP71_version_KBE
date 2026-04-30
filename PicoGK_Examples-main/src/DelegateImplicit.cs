using System;
using System.Numerics;
using PicoGK;

namespace Leap71.PythonInterop
{
    /// <summary>
    /// Wraps a Func&lt;Vector3, float&gt; delegate as an IImplicit so Python (via
    /// pythonnet) can provide a signed-distance function without needing to
    /// implement the C# interface directly.
    ///
    /// Usage from Python:
    ///   from Leap71.PythonInterop import DelegateImplicit
    ///   from System import Func, Single
    ///   from System.Numerics import Vector3
    ///
    ///   def my_sdf(vec):
    ///       ...
    ///       return float_value
    ///
    ///   implicit = DelegateImplicit(Func[Vector3, Single](my_sdf))
    ///   vox = Sh.voxIntersectImplicit(vox_bounding, implicit)
    /// </summary>
    public class DelegateImplicit : IImplicit
    {
        readonly Func<Vector3, float> m_fnSdf;

        public DelegateImplicit(Func<Vector3, float> fnSdf)
        {
            m_fnSdf = fnSdf;
        }

        public float fSignedDistance(in Vector3 vecPt)
        {
            return m_fnSdf(vecPt);
        }
    }
}
