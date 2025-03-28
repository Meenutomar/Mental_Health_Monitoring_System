"use client"; 
import { useState } from "react";

const templates = [
  { id: 1, name: "Modern Blue", image: "/templates/modern-blue.jpeg" },
  { id: 2, name: "Classic Black", image: "/templates/classic-black.webp" },
  { id: 3, name: "Minimalist", image: "/templates/modern-red.webp" },
  { id: 4, name: "Creative", image: "/templates/creative.webp" }
];

export default function TemplateCarousel({ onSelect }) {
    const [selected, setSelected] = useState(null);
  
    return (
      <div className="flex flex-col items-center">
        <h2 className="text-2xl font-bold my-4">Select a Resume Template</h2>
        <div className="carousel w-full max-w-lg">
          {templates.map((template, index) => (
            <div
              key={template.id}
              id={`slide${index}`}
              className="carousel-item relative w-full"
            >
              <img
                src={template.image}
                alt={template.name}
                className="rounded-lg shadow-lg border border-gray-200"
              />
              <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2">
                <button
                  className={`btn btn-sm ${
                    selected === template.id ? "btn-primary" : "btn-outline"
                  }`}
                  onClick={() => {
                    setSelected(template.id);
                    onSelect(template);
                  }}
                >
                  {selected === template.id ? "Selected" : "Select"}
                </button>
              </div>
            </div>
          ))}
        </div>
  
        <div className="flex justify-center gap-2 mt-4">
          {templates.map((_, index) => (
            <a
              key={index}
              href={`#slide${index}`}
              className="btn btn-xs btn-circle"
            ></a>
          ))}
        </div>
      </div>
    );
  }