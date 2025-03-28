"use client";

import { useEffect, useState } from "react";
import Image from "next/image";

const themes = ["light", "dark", "cupcake", "bumblebee", "emerald", "synthwave", "halloween"];

const Header = ({ user, onLogout }) => {
  const [theme, setTheme] = useState(null);

  useEffect(() => {
    const storedTheme = localStorage.getItem("theme") || "light";
    setTheme(storedTheme);
    document.documentElement.setAttribute("data-theme", storedTheme);
  }, []);

  const handleThemeChange = (t) => {
    setTheme(t);
    localStorage.setItem("theme", t);
    document.documentElement.setAttribute("data-theme", t);
  };

  if (theme === null) return null; // avoid hydration mismatch

  return (
    <header className="bg-primary text-white p-2 shadow-md flex justify-between items-center flex-wrap gap-2">
      {/* Logo & Title */}
      <div className="flex gap-4 items-center">
        <a href="/" className="flex items-center gap-2">
          <Image src="/logo.png" alt="RoboMH Logo" width={40} height={40} />
          <h1 className="text-xl font-bold">RoboMH</h1>
        </a>
      </div>

      {/* Navigation */}
      <nav>
        <ul className="flex space-x-4">
          <li>
            <a href="/" className="hover:underline">Home</a>
          </li>
          <li>
            <a href="/templates" className="hover:underline">Templates</a>
          </li>
          <li>
            <a href="/contact" className="hover:underline">Support</a>
          </li>
        </ul>
      </nav>

      {/* Right Side: Welcome + Theme + Logout */}
      <div className="flex items-center gap-4">
        {/* Theme Switcher */}
        <div className="dropdown dropdown-end">
          <label tabIndex={0} className="btn btn-sm btn-outline text-white">
            🎨 Theme
          </label>
          <ul tabIndex={0} className="dropdown-content menu p-2 shadow bg-base-100 rounded-box w-40 z-50">
            {themes.map((t) => (
              <li key={t}>
                <button
                  className={`btn btn-sm w-full ${theme === t ? "btn-active" : ""}`}
                  onClick={() => handleThemeChange(t)}
                >
                  {t.charAt(0).toUpperCase() + t.slice(1)}
                </button>
              </li>
            ))}
          </ul>
        </div>

        {/* Welcome Message + Logout */}
        {user && (
          <div className="flex items-center gap-2">
            <span className="text-sm hidden sm:inline">
              Welcome, {user.user_metadata?.name || user.email}
            </span>
            <button className="btn btn-sm btn-secondary" onClick={onLogout}>
              Logout
            </button>
          </div>
        )}
      </div>
    </header>
  );
};

export default Header;
