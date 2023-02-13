using System;
using System.Runtime.InteropServices;
namespace DynamicLibraries
{
    public static class DynamicLibWrapper
    {
        private const string DLLName = "MachineLearningLib";

        [DllImport(DLLName)]
        public static extern int test();

        [DllImport(DLLName)]
        public static extern void CreateModelPMC(IntPtr npl, int sizeNpl, int maxN, IntPtr X, IntPtr deltas, IntPtr W);
    }
}