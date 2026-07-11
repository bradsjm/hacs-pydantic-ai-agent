import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    'intro', 'quickstart',
    {type: 'category', label: 'Installation', items: ['installation/requirements', 'installation/hacs', 'installation/manual']},
    {type: 'category', label: 'Concepts', items: ['concepts/workspaces-subentries', 'concepts/providers-profiles', 'concepts/agents-tools', 'concepts/history-context']},
    {type: 'category', label: 'Configuration', items: ['configuration/providers', 'configuration/conversation', 'configuration/ai-tasks', 'configuration/mcp', 'configuration/skills', 'configuration/context-web-workspace', 'configuration/observability']},
    {type: 'category', label: 'Reference', items: ['reference/services', 'reference/entities', 'reference/diagnostics', 'reference/defaults']},
    {type: 'category', label: 'Operations', items: ['operations/privacy-security', 'operations/troubleshooting', 'operations/upgrades']},
    {type: 'category', label: 'Development', items: ['development/architecture', 'development/local-setup', 'development/validation', 'development/contributing']},
    'release-notes',
  ],
};

export default sidebars;
