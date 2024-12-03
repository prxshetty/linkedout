import Link from 'next/link';

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900">
            Resume Optimizer
          </h1>
          <p className="mt-3 max-w-2xl mx-auto text-xl text-gray-500">
            Optimize your resume for job applications using AI
          </p>
          <div className="mt-5 flex justify-center gap-4">
            <Link 
              href="/jobs/analyze"
              className="px-4 py-2 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700"
            >
              Analyze Job
            </Link>
            <Link
              href="/resumes/optimize"
              className="px-4 py-2 border border-transparent text-base font-medium rounded-md text-white bg-green-600 hover:bg-green-700"
            >
              Optimize Resume
            </Link>
          </div>
        </div>
      </div>
    </main>
  );
}