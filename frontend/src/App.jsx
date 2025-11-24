import React, { useState } from 'react';
import InputSection from './components/InputSection';
import ResultsTable from './components/ResultsTable';
import { analyzeCase } from './api';
import { Scale } from 'lucide-react';

function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleAnalyze = async (text) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await analyzeCase(text);
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen text-gray-100 font-sans selection:bg-blue-500/30">
      {/* Header */}
      <header className="pt-12 pb-8 text-center">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="p-3 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-lg shadow-blue-500/20">
            <Scale className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Presidio AI
          </h1>
        </div>
        <p className="text-gray-400 text-lg max-w-2xl mx-auto px-4">
          Advanced legal case similarity search powered by AI.
          Compare facts and judgements across multiple models.
        </p>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 pb-20">
        <InputSection onAnalyze={handleAnalyze} isLoading={isLoading} />

        {error && (
          <div className="max-w-4xl mx-auto mb-8 p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-center animate-fade-in">
            Error: {error}
          </div>
        )}

        <ResultsTable results={results} />
      </main>
    </div>
  );
}

export default App;
