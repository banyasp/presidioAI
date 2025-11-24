import React, { useState } from 'react';
import { Search, Loader2 } from 'lucide-react';

const InputSection = ({ onAnalyze, isLoading }) => {
  const [text, setText] = useState('');
  const [model, setModel] = useState('legal-bert');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim()) {
      onAnalyze(text, model);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto mb-12 animate-fade-in">
      <form onSubmit={handleSubmit} className="flex flex-col gap-4">
        <div className="flex justify-between items-center mb-2">
          <label className="text-sm font-medium text-gray-400">Select AI Model:</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="bg-gray-800 text-gray-200 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block p-2.5 border border-gray-700"
            disabled={isLoading}
          >
            <option value="legal-bert">Legal BERT (Default)</option>
            <option value="harvard-bert">Harvard BERT (CaseHOLD)</option>
            <option value="sentence-transformer">Sentence Transformer (MiniLM)</option>
          </select>
        </div>

        <div className="relative group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl opacity-30 group-hover:opacity-75 transition duration-500 blur"></div>
          <div className="relative bg-gray-900 rounded-xl p-1">
            <textarea
              className="w-full h-48 bg-gray-900 text-gray-100 p-6 rounded-lg focus:outline-none resize-none text-lg leading-relaxed placeholder-gray-500 font-light"
              placeholder="Paste legal text here to find similar cases..."
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
        </div>
        
        <div className="flex justify-end">
          <button
            type="button"
            onClick={() => setText("The defendant was denied counsel during the trial.")}
            className="mr-auto text-gray-400 hover:text-white text-sm transition-colors"
          >
            Try Example
          </button>

          <button
            type="submit"
            disabled={isLoading || !text.trim()}
            className={`
              flex items-center gap-2 px-8 py-3 rounded-lg font-medium text-white transition-all duration-300
              ${isLoading || !text.trim() 
                ? 'bg-gray-700 cursor-not-allowed opacity-50' 
                : 'bg-gradient-to-r from-blue-600 to-purple-600 hover:shadow-lg hover:shadow-blue-500/30 transform hover:-translate-y-0.5'}
            `}
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Search className="w-5 h-5" />
                Find Similar Cases
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default InputSection;
