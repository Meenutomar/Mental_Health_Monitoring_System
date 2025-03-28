const Footer = () => {
  return (
    <footer className="footer footer-center p-4 bg-base-300 text-neutral-content mt-10">
      <div>
        <p>&copy; {new Date().getFullYear()} RoboMH - AI powered mental health bot. All rights reserved.</p>
      </div>

      {/* Optional Social Media Links */}
      <nav className="grid grid-flow-col gap-4">
        <a href="https://github.com/your-profile" target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-sm">
          <svg className="w-5 h-5 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M12 .5C5.4.5 0 5.9 0 12.5c0 5.3 3.4 9.7 8.2 11.3.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.5-1.2-1.2-1.5-1.2-1.5-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 .1 1.6-.2 2-.4.1-.7.4-1.2.7-1.4-2.6-.3-5.4-1.3-5.4-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.3 11.3 0 0 1 6 0c2.2-1.5 3.3-1.2 3.3-1.2.6 1.6.2 2.8.1 3.1.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.4 5.9.5.4.8 1 .8 2v2.9c0 .3.2.7.8.6 4.8-1.6 8.2-6 8.2-11.3C24 5.9 18.6.5 12 .5z" />
          </svg>
        </a>

        <a href="https://linkedin.com/in/your-profile" target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-sm">
          <svg className="w-5 h-5 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M4.98 3.5a2.2 2.2 0 1 0 0 4.4 2.2 2.2 0 0 0 0-4.4zm-.02 5.5H2v12h3V9zm5-1c-1.5 0-3 .8-3 3v10h3v-6c0-1.1.9-2 2-2h1v-3h-1c-.7 0-1 .3-1 1zm11 1v-1h-1c-.7 0-1 .3-1 1v1h-3v3h3v6h3V12h1l1-3h-2z" />
          </svg>
        </a>

        <a href="https://twitter.com/your-profile" target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-sm">
          <svg className="w-5 h-5 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
            <path d="M24 4.6a9.8 9.8 0 0 1-2.8.8 4.8 4.8 0 0 0 2.1-2.7 9.7 9.7 0 0 1-3.1 1.2 4.8 4.8 0 0 0-8.2 4.4A13.7 13.7 0 0 1 1.7 3.1 4.8 4.8 0 0 0 3 9a4.8 4.8 0 0 1-2.2-.6v.1a4.8 4.8 0 0 0 3.8 4.7 4.7 4.7 0 0 1-2.1.1 4.8 4.8 0 0 0 4.5 3.3A9.6 9.6 0 0 1 0 19.5a13.7 13.7 0 0 0 7.5 2.2 13.7 13.7 0 0 0 13.7-13.7c0-.2 0-.4-.1-.6a9.8 9.8 0 0 0 2.4-2.5z" />
          </svg>
        </a>
      </nav>
    </footer>
  );
};

export default Footer;
