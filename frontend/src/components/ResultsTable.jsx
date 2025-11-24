import React, { useState } from 'react';
import { Scale, FileText, Gavel, ChevronDown, ChevronUp } from 'lucide-react';

const ResultsTable = ({ results }) => {
  // Track expanded state for each result's sections
  const [expandedSections, setExpandedSections] = useState({});

  const toggleSection = (resultIndex, section) => {
    const key = `${resultIndex}-${section}`;
    setExpandedSections(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const isExpanded = (resultIndex, section) => {
    const key = `${resultIndex}-${section}`;
    return expandedSections[key] || false;
  };

  if (!results || results.length === 0) return null;

  return (
    <div className="w-full max-w-6xl mx-auto space-y-12 animate-fade-in-up">
      {results.map((result, index) => (
        <div key={index} className="bg-gray-900/50 backdrop-blur-xl border border-gray-800 rounded-2xl overflow-hidden shadow-2xl">
          {/* Header */}
          <div className="bg-gray-800/50 p-6 border-b border-gray-700 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-blue-500/10 rounded-lg">
                <Scale className="w-6 h-6 text-blue-400" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white">{result.case_name}</h3>
                <p className="text-sm text-gray-400">Similarity Score: {(result.similarity_score * 100).toFixed(1)}%</p>
              </div>
            </div>
          </div>

          {/* Comparison Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-gray-800 bg-gray-900/80">
                  <th className="p-6 text-xs font-bold text-gray-500 uppercase tracking-wider w-1/4">Section</th>
                  <th className="p-6 text-xs font-bold text-gray-500 uppercase tracking-wider w-3/4 border-l border-gray-800">
                    <span className="text-blue-400">Database Source</span> (Retrieval)
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {/* Case Facts Row */}
                <tr className="group hover:bg-gray-800/30 transition-colors">
                  <td className="p-6 align-top">
                    <div className="flex items-center gap-2 text-gray-300 font-medium">
                      <FileText className="w-4 h-4" />
                      Case Facts
                    </div>
                  </td>
                  <td className="p-6 align-top border-l border-gray-800">
                    <button
                      onClick={() => toggleSection(index, 'facts')}
                      className="w-full flex items-center justify-between text-left text-gray-300 hover:text-white transition-colors group/button"
                    >
                      <span className="font-medium">
                        {isExpanded(index, 'facts') ? 'Hide Details' : 'Show Details'}
                      </span>
                      {isExpanded(index, 'facts') ? (
                        <ChevronUp className="w-5 h-5 text-blue-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-blue-400" />
                      )}
                    </button>
                    <div
                      className={`overflow-hidden transition-all duration-300 ease-in-out ${
                        isExpanded(index, 'facts') ? 'max-h-[2000px] opacity-100 mt-4' : 'max-h-0 opacity-0'
                      }`}
                    >
                      <div className="text-gray-300 leading-relaxed">
                        {result.case_facts["Database Source"]}
                      </div>
                    </div>
                  </td>
                </tr>

                {/* Judgement Row */}
                <tr className="group hover:bg-gray-800/30 transition-colors">
                  <td className="p-6 align-top">
                    <div className="flex items-center gap-2 text-gray-300 font-medium">
                      <Gavel className="w-4 h-4" />
                      Judgement
                    </div>
                  </td>
                  <td className="p-6 align-top border-l border-gray-800">
                    <button
                      onClick={() => toggleSection(index, 'judgement')}
                      className="w-full flex items-center justify-between text-left text-gray-300 hover:text-white transition-colors group/button"
                    >
                      <span className="font-medium">
                        {isExpanded(index, 'judgement') ? 'Hide Details' : 'Show Details'}
                      </span>
                      {isExpanded(index, 'judgement') ? (
                        <ChevronUp className="w-5 h-5 text-blue-400" />
                      ) : (
                        <ChevronDown className="w-5 h-5 text-blue-400" />
                      )}
                    </button>
                    <div
                      className={`overflow-hidden transition-all duration-300 ease-in-out ${
                        isExpanded(index, 'judgement') ? 'max-h-[2000px] opacity-100 mt-4' : 'max-h-0 opacity-0'
                      }`}
                    >
                      <div className="text-gray-300 leading-relaxed">
                        {result.judgement["Database Source"]}
                      </div>
                    </div>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ))}
    </div>
  );
};

export default ResultsTable;
