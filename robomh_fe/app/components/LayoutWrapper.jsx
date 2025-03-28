import { useUser, useSupabaseClient } from "@supabase/auth-helpers-react";

export default function LayoutWrapper({ children }) {
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const supabase = useSupabaseClient();
  const { user } = useUser();

  const handleLogout = async () => {
    await supabase.auth.signOut();
  };

  return (
    <div className={`drawer ${isSidebarOpen ? "drawer-open" : ""}`}>
      <input
        id="sidebar-drawer"
        type="checkbox"
        className="drawer-toggle"
        checked={isSidebarOpen}
        onChange={() => setIsSidebarOpen(!isSidebarOpen)}
      />
      <div className="drawer-content flex flex-col">
        <Header user={user} onLogout={handleLogout} />
        <button
          className="btn btn-primary m-4 lg:hidden"
          onClick={() => setIsSidebarOpen(!isSidebarOpen)}
        >
          {isSidebarOpen ? "Close Menu" : "Open Menu"}
        </button>
        <main className="container mx-auto p-4 flex-grow">{children}</main>
        <Footer />
      </div>

      <div className="drawer-side">
        <label
          htmlFor="sidebar-drawer"
          className="drawer-overlay"
          onClick={() => setIsSidebarOpen(false)}
        ></label>
        <Sidebar closeSidebar={() => setIsSidebarOpen(false)} />
      </div>
    </div>
  );
}
