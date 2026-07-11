import type {Config} from '@docusaurus/types';
import type {Preset} from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Pydantic AI Agent',
  tagline: 'Pydantic AI-powered conversation agents and AI tasks for Home Assistant',
  url: 'https://bradsjm.github.io',
  baseUrl: '/hacs-pydantic-ai-agent/',
  organizationName: 'bradsjm',
  projectName: 'hacs-pydantic-ai-agent',
  trailingSlash: false,
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'throw',
  i18n: {defaultLocale: 'en', locales: ['en']},
  presets: [[
    'classic',
    {
      docs: {
        routeBasePath: '/',
        sidebarPath: './sidebars.ts',
        editUrl: 'https://github.com/bradsjm/hacs-pydantic-ai-agent/edit/main/docs-site/',
      },
      blog: false,
      theme: {customCss: './src/css/custom.css'},
    } satisfies Preset.Options,
  ]],
  themeConfig: {
    navbar: {
      title: 'Pydantic AI Agent',
      items: [
        {type: 'docSidebar', sidebarId: 'docsSidebar', position: 'left', label: 'Docs'},
        {href: 'https://github.com/bradsjm/hacs-pydantic-ai-agent', label: 'GitHub', position: 'right'},
        {href: 'https://github.com/bradsjm/hacs-pydantic-ai-agent/issues', label: 'Issues', position: 'right'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {title: 'Get started', items: [{label: 'Installation', to: '/installation/requirements'}, {label: 'Quickstart', to: '/quickstart'}, {label: 'Troubleshooting', to: '/operations/troubleshooting'}]},
        {title: 'Reference', items: [{label: 'Services', to: '/reference/services'}, {label: 'Defaults', to: '/reference/defaults'}, {label: 'Diagnostics', to: '/reference/diagnostics'}]},
        {title: 'Project', items: [{label: 'GitHub', href: 'https://github.com/bradsjm/hacs-pydantic-ai-agent'}, {label: 'Releases', to: '/release-notes'}]},
      ],
      copyright: `Copyright ${new Date().getFullYear()} Pydantic AI Agent contributors.`,
    },
    prism: {additionalLanguages: ['python', 'yaml', 'json']},
  } satisfies Preset.ThemeConfig,
};

export default config;
