import Image from "next/image";
import Link from "next/link";

const Sidebar = ({ closeSidebar }) => {
  return (
    <aside className="bg-base-200 shadow-xl h-full w-64 p-4 flex flex-col justify-between">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3 mb-6 px-2">
          <Image src="/logo.png" alt="RoboMH Logo" width={40} height={40} />
          <h1 className="text-lg font-bold leading-tight">RoboMH</h1>
        </div>
        <p className="text-sm text-gray-500 px-2 mb-4">
          AI-powered Mental Health Assistant
        </p>

        {/* Navigation */}
        <nav>
          <ul className="menu p-0">
            <li className="menu-title px-2 text-xs uppercase text-gray-400">
              Sessions
            </li>
            <li>
              <Link href="/" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                💬 Chat
              </Link>
            </li>
            <li>
              <Link href="/audio" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                🔊 Audio
              </Link>
            </li>
            <li>
              <Link href="/video" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                🎥 Video
              </Link>
            </li>

            <li className="menu-title px-2 mt-4 text-xs uppercase text-gray-400">
              Account
            </li>
            <li>
              <Link href="/builder" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                👤 My Profile
              </Link>
            </li>
            <li>
              <Link href="/sessions" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                📑 My Sessions
              </Link>
            </li>
            <li>
              <Link href="/reports" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                📊 My Reports
              </Link>
            </li>
            <li>
              <Link href="/settings" onClick={closeSidebar} className="hover:bg-primary hover:text-white rounded-lg">
                ⚙️ Settings
              </Link>
            </li>
          </ul>
        </nav>
      </div>

      {/* Optional footer */}
      <div className="text-xs text-center text-gray-400 mt-4">
        <p>© 2025 RoboMH</p>
      </div>
    </aside>
  );
};

export default Sidebar;
